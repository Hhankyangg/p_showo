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

conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]

import os
from omegaconf import DictConfig, ListConfig, OmegaConf

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def prepare_inputs_and_labels(
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


config = OmegaConf.load('configs/showo_demo.yaml')
device = torch.device("cuda:2")
data_root = "/home/hpyky/full_mcdata"
concept = "dunpai"
nums_new_token_i = 16
save_path = os.path.join("saves", concept, "only_t2i_wo_sys")


tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side ="left")
uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)
vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name).to(device)
model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(device)


#################################
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

with torch.no_grad():
    orig_embeds = model.showo.get_input_embeddings().weight.data.clone()
    orig_lm_head_weight = model.showo.lm_head.weight.data.clone()
    orig_lm_head_bias = model.showo.lm_head.bias.data.clone()
    

vq_model.requires_grad_ = False
vq_model.eval()
model.train()
for names, p in model.named_parameters():
    if "embed_tokens" not in names and "lm_head" not in names:
        p.requires_grad = False
    else:
        p.requires_grad = True

trainable_params = [model.showo.get_input_embeddings().weight, model.showo.lm_head.weight, model.showo.lm_head.bias]
optimizer = torch.optim.AdamW(
            trainable_params, # for optimize the embeddings and the head
            lr=1e-2,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-08,
        )
for names, p in model.named_parameters():
    if p.requires_grad:
        print(f"{names} requires_grad") # embed_token, lm_head会更新
        

model.config.mask_token_id = model.showo.get_input_embeddings().num_embeddings - 1
model.mask_token_id = model.showo.get_input_embeddings().num_embeddings - 1
mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
mask_id = model.mask_token_id
mask_dtype = model.showo.model.embed_tokens.weight.dtype


t2i_dataloader = get_personalized_t2i_dataloader(data_root, concept, tokenizer, batch_size=2, num_workers=10, max_length=128)
model.output_size = new_total_vocab
os.makedirs(save_path, exist_ok=True)
global_loss_list = []
for epoch in range(0, 1000):
    print(f"Epoch {epoch+1}")
    loss_list = []
    loss_t2i_list = []
    for batch in tqdm(t2i_dataloader):
        batch_size_t2i = batch["images"].shape[0]
        
        # t2i format
        pixel_values, texts = batch["images"], batch["conditions"]
        pixel_values = pixel_values.to(device)
        input_ids, labels, mask_prob, image_tokens_ori = prepare_inputs_and_labels(pixel_values, texts, is_train=True)
        attention_mask = create_attention_mask_predict_next(input_ids,
                                                                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                rm_pad_in_image=True,
                                                                return_inverse_mask=True)
        attention_mask = attention_mask.to(mask_dtype)
        
        optimizer.zero_grad()
        
        logits, loss_t2i, loss_lm, loss_mmu = model(
                    input_ids=input_ids,
                    input_embeddings=None,
                    attention_mask=attention_mask,
                    labels=labels,
                    label_smoothing=0.0,
                    batch_size_t2i=batch_size_t2i,
                    batch_size_lm=0,
                    batch_size_mmu=0,
                    max_seq_length=128,
                )
        loss = loss_t2i
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        loss_t2i_list.append(loss_t2i.item())

        # 恢复原始权重
        with torch.no_grad():
            model.showo.get_input_embeddings().weight.data[index_no_updates] = orig_embeds[index_no_updates]
            model.showo.lm_head.weight.data[index_no_updates] = orig_lm_head_weight[index_no_updates]
            model.showo.lm_head.bias.data[index_no_updates] = orig_lm_head_bias[index_no_updates]
    print(f"Epoch {epoch+1} loss: {np.mean(loss_list)}, loss_t2i: {np.mean(loss_t2i_list)}")
    print(f"  Token-Norm: {model.showo.get_input_embeddings().weight[new_token_ids].norm().item()}")
    print(f"  index_no_updates-Token-Norm: {model.showo.get_input_embeddings().weight[index_no_updates].norm().item()}")
    print(f"  LM-Head-Weight-Norm: {model.showo.lm_head.weight[new_token_ids].norm().item()}")
    print(f"  index_no_updates-LM-Head-Weight-Norm: {model.showo.lm_head.weight[index_no_updates].norm().item()}")
    print(f"  LM-Head-Bias-Norm: {model.showo.lm_head.bias[new_token_ids].norm().item()}")
    print(f"  index_no_updates-LM-Head-Bias-Norm: {model.showo.lm_head.bias[index_no_updates].norm().item()}")
    global_loss_list.append(np.mean(loss_list))
    
    if (epoch+1) % 100 == 0:
        save_path_embed = os.path.join(save_path, f"epoch_{epoch+1}_embed.pt")
        save_path_lm_head_weight = os.path.join(save_path, f"epoch_{epoch+1}_lm_head_weight.pt")
        save_path_lm_head_bias = os.path.join(save_path, f"epoch_{epoch+1}_lm_head_bias.pt")
        
        torch.save(model.showo.get_input_embeddings().weight.data[new_token_ids], save_path_embed)
        torch.save(model.showo.lm_head.weight.data[new_token_ids], save_path_lm_head_weight)
        torch.save(model.showo.lm_head.bias.data[new_token_ids], save_path_lm_head_bias)
        