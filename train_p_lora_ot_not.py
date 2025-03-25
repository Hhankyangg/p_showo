import argparse
from typing import Union
from pdata import PersonalizedMMUDataset, PersonalizedT2IDataset, get_personalized_mmu_dataloader, get_personalized_t2i_dataloader
from lightning.pytorch.utilities import CombinedLoader

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image

from models import Showo, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next, create_attention_mask_for_mmu
from training.utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter
from transformers import AutoTokenizer
from models.clip_encoder import CLIPVisionTower
from transformers import CLIPImageProcessor
from llava.llava import conversation as conversation_lib
from peft import LoraConfig, get_peft_model

conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]

import os
from omegaconf import DictConfig, ListConfig, OmegaConf


def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/showo_demo.yaml")
    parser.add_argument("--data_root", type=str, default="/home/hpyky/full_mcdata")
    
    parser.add_argument("--concept", type=str, default="dunpai")
    parser.add_argument("--task_name", type=str, default="test")
    
    parser.add_argument("--need_new_tokens", default=False, action="store_true")
    parser.add_argument("--need_lora", default=False, action="store_true")
    parser.add_argument("--t2i_data", default=False, action="store_true")
    parser.add_argument("--mmu_data", default=False, action="store_true")
    
    parser.add_argument("--epoch", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--nums_new_token_i", type=int, default=16)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", nargs='+', default=["fc1", "k_proj", "v_proj", "q_proj", "fc2"])
    return parser.parse_args()


def setup_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side ="left")
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)
    vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name).to("cuda")
    model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to("cuda")
    return tokenizer, uni_prompting, vq_model, model


def update_tokens(concept, tokenizer, model, nums_new_token_i=16):
    new_tokens = [f"<{concept}>"] + [f"<token_{i}>" for i in range(nums_new_token_i)]
    num_new_tokens = len(new_tokens)  # 17
    # 已知的原始参数
    # 文本 token 数量（ID 0-50304）
    original_text_vocab_size = len(tokenizer) 
    # Image token 数量（原 ID 50305-58497）
    original_image_vocab_size = model.showo.get_input_embeddings().num_embeddings - len(tokenizer)

    original_total_vocab = original_text_vocab_size + original_image_vocab_size  # 58498
    
    # 新的参数
    new_text_vocab_size = original_text_vocab_size + num_new_tokens  # 50305 + 17 = 50322
    new_total_vocab = original_total_vocab + num_new_tokens          # 58498 + 17 = 58515

    # ------------------------------
    # Step 1: 修改 Tokenizer 的词汇表
    # ------------------------------

    # 添加新 token 到 50305-50321 的位置
    num_new_tokens = tokenizer.add_tokens(new_tokens)
    new_token_ids = tokenizer.convert_tokens_to_ids(new_tokens)
    print("新 token ID:", new_token_ids)  # 应输出 50305-50321
    
    # ------------------------------
    # Step 2: 调整模型的权重
    # ------------------------------
    with torch.no_grad():
        # 获取嵌入层权重
        embeddings = model.showo.get_input_embeddings().weight.data
        
        # 扩展嵌入层（58498 -> 58515）
        model.showo.resize_token_embeddings(new_total_vocab)
        # new_embeddings = model.showo.get_input_embeddings().weight.data

        # 将原 Image Token 权重后移 17 位
        original_image_weights = embeddings[original_text_vocab_size:original_total_vocab].clone()
        model.showo.get_input_embeddings().weight.data[new_text_vocab_size:new_total_vocab] = original_image_weights
        
        # 初始化新 token 的权重（用原文本最后 17 个 token）
        # new_text_weights = embeddings[original_text_vocab_size - num_new_tokens : original_text_vocab_size].clone()
        # model.showo.get_input_embeddings().weight.data[original_text_vocab_size : new_text_vocab_size] = new_text_weights
        # print(model.showo.lm_head.weight.data.shape[1])
        # 处理 lm_head（假设与嵌入层共享权重）
        if model.showo.lm_head.weight.data.shape[0] == new_total_vocab:
            # 扩展 lm_head 权重
            lm_head = model.showo.lm_head
            new_lm_head = torch.nn.Linear(
                lm_head.in_features, 
                new_total_vocab, 
                bias=hasattr(lm_head, 'bias')
            )
            new_lm_head.weight.data = lm_head.weight.data.clone()
            new_lm_head.weight.data[new_text_vocab_size:new_total_vocab] = lm_head.weight.data[original_text_vocab_size:original_total_vocab]
            # new_lm_head.weight.data[original_text_vocab_size:new_text_vocab_size] = lm_head.weight.data[original_text_vocab_size - num_new_tokens : original_text_vocab_size]
            if hasattr(lm_head, 'bias'):
                new_lm_head.bias.data = lm_head.bias.data.clone()
                new_lm_head.bias.data[new_text_vocab_size:new_total_vocab] = lm_head.bias.data[original_text_vocab_size:original_total_vocab]
                # new_lm_head.bias.data[original_text_vocab_size:new_text_vocab_size] = lm_head.bias.data[original_text_vocab_size - num_new_tokens : original_text_vocab_size]
            
            model.showo.lm_head = new_lm_head
        else:
            raise ValueError("lm_head weights do not match the input embeddings!")

    index_no_updates = torch.ones((new_total_vocab,), dtype=torch.bool)
    index_no_updates[new_token_ids] = False
    
    # ------------------------------
    # 验证
    # ------------------------------
    # 检查新 token 的 ID
    print("新增文本 token ID:", [tokenizer.convert_tokens_to_ids(t) for t in new_tokens])  # 应输出 50305-50321

    # 检查一个原 Image Token 的新 ID
    sample_image_token = tokenizer.convert_ids_to_tokens(original_text_vocab_size)  # 原 ID 50305
    print(f"Concept Token '{sample_image_token}' 的新 ID:", tokenizer.convert_tokens_to_ids(sample_image_token))  # 应输出 50322

    # 检查嵌入层形状
    print("嵌入层大小:", model.showo.get_input_embeddings().weight.shape)  # 应显示 torch.Size([58515, 2048])

    # 检查 index_no_updates 中 True 的位置和数量，True 应该是 new token ids
    print("index_no_updates 中 False 的位置:", torch.nonzero(~index_no_updates).squeeze())  # 应输出 50305-50321
    print("index_no_updates 中 True 的数量:", torch.sum(index_no_updates))  # 应输出 58498

    with torch.no_grad():
        orig_embeds = model.showo.get_input_embeddings().weight.data.clone()
        orig_lm_head_weight = model.showo.lm_head.weight.data.clone()
        orig_lm_head_bias = model.showo.lm_head.bias.data.clone()
        
    return tokenizer, model, orig_embeds, orig_lm_head_weight, orig_lm_head_bias, index_no_updates, new_total_vocab, new_token_ids


def apply_lora(model, args):
    lora_config = LoraConfig(
        r=args.lora_r,  # LoRA 层的 r 参数  
        lora_alpha=args.lora_alpha,  # LoRA 层的 alpha 参数
        lora_dropout=args.lora_dropout,  # LoRA层的dropout
        task_type="CAUSAL_LM",  # 任务类型为因果语言模型
        target_modules = args.lora_target_modules  # 需要应用 LoRA 的模块
    )
    
    model.showo = get_peft_model(model.showo, lora_config)
    
    return model


def prepare_inputs_and_labels(
        mask_id,
        config,
        vq_model,
        uni_prompting,
        mask_schedule,
        pixel_values_or_image_ids: Union[torch.FloatTensor, torch.LongTensor],
        texts: Union[str, str],
        min_masking_rate: float = 0.0,
        is_train: bool = True,
):

    image_tokens = vq_model.get_code(pixel_values_or_image_ids)
    image_tokens = image_tokens + len(uni_prompting.text_tokenizer)

    # create MLM mask and labels
    input_ids, labels, loss_weight, mask_prob = mask_or_random_replace_tokens(
        image_tokens,
        mask_id,
        config,
        mask_schedule=mask_schedule,
        is_train=is_train,
    )
    input_ids, masks, labels = uni_prompting((texts, input_ids, labels), 't2i')

    return input_ids, labels, mask_prob, image_tokens


def main():
    args = get_test_args()
    
    config = OmegaConf.load(args.config_file)
    tokenizer, uni_prompting, vq_model, model = setup_model(config)
    
    data_root = args.data_root
    concept = args.concept
    save_path = os.path.join("saves", concept, args.task_name)
    os.makedirs(save_path, exist_ok=True)
    
    # set up training arch
    if args.need_new_tokens:
        tokenizer, model, orig_embeds, orig_lm_head_weight, orig_lm_head_bias, index_no_updates, new_total_vocab, new_token_ids = update_tokens(concept, tokenizer, model, args.nums_new_token_i)
    
    if args.need_lora:
        model = apply_lora(model, args)
    
    # set up parameters
    vq_model.requires_grad_ = False
    vq_model.eval()
    model.train()

    for name, param in model.named_parameters():
        
        
        if args.need_lora and args.need_new_tokens:
            if "lora" in name or "embed_tokens" in name or "lm_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        elif args.need_lora:
            if "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        elif args.need_new_tokens:
            if "embed_tokens" in name or "lm_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)

    optimizer = torch.optim.AdamW(
                trainable_params, # for optimize the embeddings and the head
                lr=args.lr,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
                eps=1e-08,
            )
    
    for names, p in model.named_parameters():
        if p.requires_grad:
            print(f"{names} requires_grad") # embed_token, lm_head会更新
            
    #统计名称含有 "lora" 的模块的可训练参数数量
    lora_params = list(filter(lambda kv: "lora" in kv[0], model.named_parameters()))
    lora_params_num = sum(p.numel() for n, p in lora_params)
    print(f"LoRA parameters: {lora_params_num}")
    # LoRA: Q, V, mlp.fc1, mlp.fc2
    # token 可训练参数 2048*58515*2 + 58515 = 239735955
    # 统计所有可训练参数数量
    trainable_params_num = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {trainable_params_num}")
    
    # set up dataset
    if args.t2i_data:
        t2i_dataloader = get_personalized_t2i_dataloader(data_root, concept, tokenizer, batch_size=2, num_workers=0, max_length=128)
    if args.mmu_data:
        if args.need_new_tokens:
            mmu_dataloader = get_personalized_mmu_dataloader(data_root, concept, tokenizer, batch_size=5, num_workers=0, max_length=128, new_tokens=True)
        else:
            mmu_dataloader = get_personalized_mmu_dataloader(data_root, concept, tokenizer, batch_size=5, num_workers=0, max_length=128, new_tokens=False)
        
    if args.t2i_data and args.mmu_data:
        iterables = {
            'mmu_flow': mmu_dataloader,
            't2i_flow': t2i_dataloader
        }
        combined_dataloader = CombinedLoader(iterables, mode="max_size_cycle")
    elif args.t2i_data:
        iterables = {
            't2i_flow': t2i_dataloader
        }
        combined_dataloader = CombinedLoader(iterables, mode="max_size_cycle")
    elif args.mmu_data:
        iterables = {
            'mmu_flow': mmu_dataloader
        }
        combined_dataloader = CombinedLoader(iterables, mode="max_size_cycle")
    else:
        raise ValueError("No dataset loaded")
    
    combined_dataloader_list = list(combined_dataloader)

    # misc setting
    model.config.mask_token_id = model.showo.get_input_embeddings().num_embeddings - 1
    model.mask_token_id = model.showo.get_input_embeddings().num_embeddings - 1
    mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
    mask_id = model.mask_token_id
    if args.need_lora:
        mask_dtype = model.showo.base_model.model.model.embed_tokens.weight.dtype
    else:
        mask_dtype = model.showo.get_input_embeddings().weight.dtype
    if args.need_new_tokens:
        model.output_size = new_total_vocab
    
    # start training
    for epoch in range(args.epoch):
        print(f"Epoch {epoch+1}")
        loss_list = []
        if args.t2i_data:
            loss_t2i_list = []
        if args.mmu_data:
            loss_mmu_list = []
        for batch, batch_idx, dataloader_idx in tqdm(combined_dataloader_list):
            if args.t2i_data:
                batch_size_t2i = batch["t2i_flow"]["images"].shape[0]
                pixel_values, texts = batch["t2i_flow"]["images"], batch["t2i_flow"]["conditions"]
                pixel_values = pixel_values.to("cuda")
                input_ids_t2i, labels_t2i, mask_prob, image_tokens_ori = prepare_inputs_and_labels(mask_id,
                                                                                        config,
                                                                                        vq_model,
                                                                                        uni_prompting,
                                                                                        mask_schedule,
                                                                                        pixel_values,
                                                                                        texts,
                                                                                        is_train=True,)
                attention_mask_t2i = create_attention_mask_predict_next(input_ids_t2i,
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True,
                                                                    return_inverse_mask=True)
                attention_mask_t2i = attention_mask_t2i.to(mask_dtype)
            if args.mmu_data:
                batch_size_mmu = batch["mmu_flow"]["images"].shape[0]
                pixel_values_mmu, input_ids_mmu, labels_mmu = (batch["mmu_flow"]["images"],
                                                            batch["mmu_flow"]["input_ids"],
                                                            batch["mmu_flow"]["labels"])
                pixel_values_mmu = pixel_values_mmu.to("cuda", non_blocking=True)
                input_ids_mmu = input_ids_mmu.to("cuda", non_blocking=True)
                image_tokens_mmu = vq_model.get_code(pixel_values_mmu)
                image_tokens_mmu = image_tokens_mmu + len(uni_prompting.text_tokenizer)
                
                input_ids_mmu = torch.cat([
                            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to("cuda"),
                            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to("cuda"),
                            image_tokens_mmu,
                            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to("cuda"),
                            input_ids_mmu,
                        ], dim=1).long()

                labels_mmu = torch.cat([
                            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to("cuda"),
                            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to("cuda"),
                            torch.ones_like(image_tokens_mmu) * uni_prompting.ignore_id,
                            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to("cuda"),
                            labels_mmu.to("cuda")
                        ], dim=1).long()
                
                
                attention_mask_mmu = create_attention_mask_for_mmu(input_ids_mmu.to("cuda"),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))
                attention_mask_mmu = attention_mask_mmu.to(mask_dtype)

            if args.t2i_data and args.mmu_data:
                attention_mask = torch.cat([attention_mask_t2i, attention_mask_mmu], dim=0)
                input_ids = torch.cat([input_ids_t2i, input_ids_mmu], dim=0)
                labels = torch.cat([labels_t2i, labels_mmu], dim=0)
            elif args.t2i_data:
                attention_mask = attention_mask_t2i
                input_ids = input_ids_t2i
                labels = labels_t2i
                batch_size_mmu = 0
            elif args.mmu_data:
                attention_mask = attention_mask_mmu
                input_ids = input_ids_mmu
                labels = labels_mmu
                batch_size_t2i = 0
            else:
                raise ValueError("No dataset loaded")
            
            optimizer.zero_grad()
            logits, loss_t2i, loss_lm, loss_mmu = model(
                        input_ids=input_ids,
                        input_embeddings=None,
                        attention_mask=attention_mask,
                        labels=labels,
                        label_smoothing=0.0,
                        batch_size_t2i=batch_size_t2i,
                        batch_size_lm=0,
                        batch_size_mmu=batch_size_mmu,
                        max_seq_length=128,
                    )
            if args.t2i_data and args.mmu_data:
                loss = 0.8 * loss_t2i + 0.2 * loss_mmu
            elif args.t2i_data:
                loss = loss_t2i
            elif args.mmu_data:
                loss = loss_mmu
            
            loss.backward()
            optimizer.step()
            
            loss_list.append(loss.item())
            if args.t2i_data:
                loss_t2i_list.append(loss_t2i.item())
            if args.mmu_data:
                loss_mmu_list.append(loss_mmu.item())
            
            if args.need_new_tokens:
                model.showo.get_input_embeddings().weight.data[index_no_updates] = orig_embeds[index_no_updates]
                model.showo.lm_head.weight.data[index_no_updates] = orig_lm_head_weight[index_no_updates]
                model.showo.lm_head.bias.data[index_no_updates] = orig_lm_head_bias[index_no_updates]

        if args.t2i_data and args.mmu_data:
            print(f"Epoch {epoch+1} loss: {np.mean(loss_list)}, loss_t2i: {np.mean(loss_t2i_list)}, loss_mmu: {np.mean(loss_mmu_list)}")
        elif args.t2i_data:
            print(f"Epoch {epoch+1} loss: {np.mean(loss_list)}, loss_t2i: {np.mean(loss_t2i_list)}")
        elif args.mmu_data:
            print(f"Epoch {epoch+1} loss: {np.mean(loss_list)}, loss_mmu: {np.mean(loss_mmu_list)}")
        else:
            raise ValueError("No dataset loaded")
        
        if (epoch+1) % 10 == 0:
            if args.need_new_tokens:
                save_path_embed = os.path.join(save_path, f"epoch_{epoch+1}_embed.pt")
                save_path_lm_head_weight = os.path.join(save_path, f"epoch_{epoch+1}_lm_head_weight.pt")
                save_path_lm_head_bias = os.path.join(save_path, f"epoch_{epoch+1}_lm_head_bias.pt")
                
                torch.save(model.showo.get_input_embeddings().weight.data[new_token_ids], save_path_embed)
                torch.save(model.showo.lm_head.weight.data[new_token_ids], save_path_lm_head_weight)
                torch.save(model.showo.lm_head.bias.data[new_token_ids], save_path_lm_head_bias)
            if args.need_lora:
                model.showo.save_pretrained(os.path.join(save_path, f"epoch_{epoch+1}_lora_model"))


if __name__ == "__main__":
    main()