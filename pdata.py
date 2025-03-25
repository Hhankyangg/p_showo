import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

import os
import json
import copy

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from functools import partial
from llava.llava import conversation as conversation_lib

# conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
DEFAULT_IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100
SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. " \
                "The assistant gives helpful, detailed, and polite answers to the user's questions."

data_root = '/home/hpyky/full_mcdata'
data_w_cases_path = os.path.join(data_root, "test_concepts.json")
with open(data_w_cases_path, "r") as f:
    data_w_cases = json.load(f)
    
    
def image_transform_dict(sample, resolution=256):
    # input image is PIL image
    image = sample["images"]
    image = image_transform(image, resolution=resolution, normalize=True)
    sample["images"] = image
    return sample

def image_transform(image, resolution=256, normalize=True):
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    return image


def preprocess_multimodal(sources):
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()

                # Customized operation, get rid of <image> special token. Edited by Zechen
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "")
                sentence['value'] = sentence['value'].strip()

    return sources


def preprocess_v0(
        sources,
        tokenizer,
):
    # Let's assume has_image is false, since we will process the image token separately
    has_image = False

    # Adapted from llava-phi/mipha/train/train.py
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence["value"])
        conversation_str = str(conv.get_prompt()).strip()
        conversations.append(conversation_str)

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "                   # ' ASSISTANT: '
    for conversation, target in zip(conversations, targets):        # loop for instances in a batch
        # total_len = int(target.ne(tokenizer.pad_token_id).sum()) + conversation.count(conv.sep2)  # in phi-2, pad_token_id == eos_token_id
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)              # handle multi-round conversation regarding one image
        cur_len = 0                                         # no bos token in phi, so set the initial len to 0
        if cur_len > 0:
            target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer(rou).input_ids) + 1  # +1 for <|endoftext|>
            instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(conversation)
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    input_ids_system = tokenizer(
        [SYSTEM_PROMPT for _ in range(len(conversations))],
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    return dict(
        input_ids=input_ids,
        labels=targets,
        input_ids_system=input_ids_system
    )
    
# class PersonalizedMMUDataset_LLAVA(Dataset):
#     def __init__(
#         self,
#         data_root: str,
#         concept_name: str,
#         clip_image_processor,
#     ):
#         self.data_root = data_root
#         self.concept_name = concept_name
#         self.clip_image_processor = clip_image_processor

#         conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
#         with open(os.path.join(data_root, f"training_data/{concept_name}.json")) as f:
#             conversations = json.load(f)
#         self.conversations = conversations

#     def __len__(self):
#         return len(self.conversations)

#     def __getitem__(self, idx):
#         conv_item = self.conversations[idx]
#         # {
#         #     "messages": [
#         #         {
#         #             "content": "<image>How would you describe <BaGu>'s attire?",
#         #             "role": "user"
#         #         },
#         #         {
#         #             "content": "The image does not provide enough information to describe <BaGu>'s attire.",
#         #             "role": "assistant"
#         #         }
#         #     ],
#         #     "images": [
#         #         "/home/hpyky/MulBench/two_concept/concept/train/BaGu/6.png"
#         #     ]
#         # }
#         image_path = conv_item["images"][0]
#         image = Image.open(image_path).convert("RGB")
#         pixel_values = self.clip_image_processor(image, return_tensors="pt")['pixel_values'][0]
        
#         question = conv_item["messages"][0]["content"].replace("<image>", "")
#         answer = conv_item["messages"][1]["content"]
        
#         conv = conversation_lib.default_conversation.copy()
#         conv.append_message(conv.roles[0], question)
#         prompt_w_o_answer = conv.get_prompt()
#         conv.append_message(conv.roles[1], answer)
#         prompt_w_answer = conv.get_prompt()
        
#         return {
#             # "image": image,
#             "images": pixel_values,   # [3, 336, 336] tensor on cpu
#             "questions": question,           # Could you confirm if this is <dunpai> in the photo?
#             "answers": answer,               # I can confirm that this is not <dunpai> in the photo.
#             "prompts_w_answer": prompt_w_answer, #  USER: Could you confirm if this is <dunpai> in the photo? ASSISTANT: I can confirm that this is not <dunpai> in the photo.<|endoftext|>
#             "prompts_w_o_answer": prompt_w_o_answer  #  USER: Could you confirm if this is <dunpai> in the photo?
#         }
        
# class PersonalizedMMUDataset(Dataset):
#     def __init__(
#         self,
#         data_root: str,
#         concept_name: str,
#     ):
#         self.data_root = data_root
#         self.concept_name = concept_name

#         conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
#         with open(os.path.join(data_root, f"training_data/{concept_name}.json")) as f:
#             conversations = json.load(f)
#         self.conversations = conversations

#     def __len__(self):
#         return len(self.conversations)

#     def __getitem__(self, idx):
#         conv_item = self.conversations[idx]
#         # {
#         #     "messages": [
#         #         {
#         #             "content": "<image>How would you describe <BaGu>'s attire?",
#         #             "role": "user"
#         #         },
#         #         {
#         #             "content": "The image does not provide enough information to describe <BaGu>'s attire.",
#         #             "role": "assistant"
#         #         }
#         #     ],
#         #     "images": [
#         #         "/home/hpyky/MulBench/two_concept/concept/train/BaGu/6.png"
#         #     ]
#         # }
#         image_path = conv_item["images"][0]
#         images = Image.open(image_path).convert("RGB")
                
#         question = conv_item["messages"][0]["content"].replace("<image>", "")
#         answer = conv_item["messages"][1]["content"]
        
#         conv = conversation_lib.default_conversation.copy()
#         conv.append_message(conv.roles[0], question)
#         prompt_w_o_answer = conv.get_prompt()
#         conv.append_message(conv.roles[1], answer)
#         prompt_w_answer = conv.get_prompt()
        
#         input_texts = prompt_w_o_answer
#         label_texts = prompt_w_answer.replace(prompt_w_o_answer, "")
        
#         item = {
#             # "image": image,
#             "images": images, 
#             "questions": question,           # Could you confirm if this is <dunpai> in the photo?
#             "answers": answer,               # I can confirm that this is not <dunpai> in the photo.
#             "prompts_w_answer": prompt_w_answer, #  USER: Could you confirm if this is <dunpai> in the photo? ASSISTANT: I can confirm that this is not <dunpai> in the photo.<|endoftext|>
#             "prompts_w_o_answer": prompt_w_o_answer,  #  USER: Could you confirm if this is <dunpai> in the photo?
#             "input_texts": input_texts,
#             "label_texts": label_texts
#         }
        
#         item = image_transform(item)
        
#         return item


# class PersonalizedMMUDataset(Dataset):
#     def __init__(
#         self,
#         data_root: str,
#         concept_name: str,
#         tokenizer,
#         max_text_len: int = 77,
#     ):
#         self.data_root = data_root
#         self.concept_name = concept_name
#         self.tokenizer = tokenizer
#         self.max_text_len = max_text_len

#         conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
        
#         # 加载数据
#         with open(os.path.join(data_root, f"training_data/{concept_name}.json")) as f:
#             conversations = json.load(f)
#         self.conversations = conversations

#     def __len__(self):
#         return len(self.conversations)

#     def __getitem__(self, idx):
#         conv_item = self.conversations[idx]

#         image_path = conv_item["images"][0]
#         image = Image.open(image_path).convert("RGB")
                
#         question = conv_item["messages"][0]["content"].replace("<image>", "")
#         answer = conv_item["messages"][1]["content"]
        
#         conv = conversation_lib.default_conversation.copy()
#         conv.append_message(conv.roles[0], question)
#         prompt_w_o_answer = conv.get_prompt()
#         conv.append_message(conv.roles[1], answer)
#         prompt_w_answer = conv.get_prompt()
        
#         input_texts = SYSTEM_PROMPT + "\n" + prompt_w_o_answer
#         label_texts = prompt_w_answer.replace(prompt_w_o_answer, "")
        
#         item = {
#             "images": image, 
#             "questions": question,           
#             "answers": answer,               
#             "prompts_w_answer": prompt_w_answer,
#             "prompts_w_o_answer": prompt_w_o_answer,
#             "input_texts": input_texts,
#             "label_texts": label_texts
#         }
        
#         item = image_transform(item)
        
#         # 使用 tokenizer 处理文本
#         input_ids = self.tokenizer(input_texts, max_length=self.max_text_len, truncation=True, padding='max_length')['input_ids']
#         label_ids = self.tokenizer(label_texts, max_length=self.max_text_len, truncation=True, padding='max_length')['input_ids']
        
#         item['input_ids'] = torch.tensor(input_ids)
#         item['label_ids'] = torch.tensor(label_ids)
        
#         # 处理 label 和 input 之间的长度关系
#         target_length = len(input_ids)
#         if len(label_ids) > target_length:
#             label_ids[target_length:] = [IGNORE_INDEX] * (len(label_ids) - target_length)
        
#         # 返回时的 item 包括了调整后的标签和输入数据
#         item['label_ids'] = torch.tensor(label_ids)
        
#         return item

# def collate_fn(instances, tokenizer=None, max_length=77):
#     input_ids = torch.stack([instance['input_ids'] for instance in instances])
#     labels = torch.stack([instance['label_ids'] for instance in instances])
    
#     # 图像处理
#     images = [instance['images'] for instance in instances]
#     images = torch.stack(images)

#     return {
#         "input_ids": input_ids,
#         "labels": labels,
#         "images": images,
#         "attention_mask": (input_ids != tokenizer.pad_token_id).long()
#     }

# def get_personalized_mmudataloader(data_root, concept_name, tokenizer, batch_size, num_workers, max_length=77):
#     dataset = PersonalizedMMUDataset(data_root, concept_name, tokenizer)
    
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         collate_fn=partial(collate_fn, tokenizer=tokenizer, max_length=max_length),
#         pin_memory=True,
#         shuffle=True 
#     )

#     return dataloader


class PersonalizedMMUDataset(Dataset):

    def __init__(self,
                 tokenizer,
                 data_root,
                 concept_name,
                 nums_new_token_i,
                 new_tokens,
                 ):
        super(PersonalizedMMUDataset, self).__init__()

        self.tokenizer = tokenizer
        
        data_file_path = os.path.join(data_root, f"showo_training_mmu_data/{concept_name}.json")

        with open(data_file_path, 'r') as f:
            data = json.load(f)
        self.list_data_dict = []
        for item in data:
            if 'image' in item.keys():
                self.list_data_dict.append(item)
        
        if new_tokens:
            self.system_personalized_prompt = f"<{concept_name}> is "
            for i in range(nums_new_token_i):
                self.system_personalized_prompt += f"<token_{i}>"
                if i == nums_new_token_i - 1:
                    self.system_personalized_prompt += "."
        else:
            self.system_personalized_prompt = ""

        for item in self.list_data_dict:
            item['conversations'][0]["value"] = self.system_personalized_prompt + "\n" + item['conversations'][0]["value"]
        
        
        print("Formatting llava instruction data")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        assert 'image' in sources[0]
        image_file = self.list_data_dict[i]['image']
        try:
            image = Image.open(os.path.join(image_file)).convert('RGB')
            image = image_transform(image)
        except:
            print("Read image error. Use dummy data.")
            crop_size = 256
            image = torch.zeros(3, crop_size, crop_size)

        sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]))

        data_dict = preprocess_v0(sources, self.tokenizer)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             input_ids_system=data_dict["input_ids_system"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        else:
            # image does not exist in the data, but the model is multimodal
            crop_size = 256
            data_dict['image'] = torch.zeros(3, crop_size, crop_size)

        return data_dict


import torch

def mmu_collate_fn(
        instances,
        tokenizer=None,
        max_length=128,
):
    """
    该函数用于将一批数据处理为一个 batch，其中所有文本（包括 input_ids 和 labels）都将填充到 max_length。
    """

    # 提取 input_ids, labels 和 input_ids_system
    input_ids, labels, input_ids_system = tuple([instance[key] for instance in instances]
                                                for key in ("input_ids", "labels", "input_ids_system"))
    
    # 填充 input_ids 和 labels 到 max_length
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id)  # 使用 pad_token_id 填充
    labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=IGNORE_INDEX)  # 使用 IGNORE_INDEX 填充 labels

    # 确保填充到 max_length
    # 如果 input_ids 长度小于 max_length，则填充
    if input_ids.shape[1] < max_length:
        pad_tube = torch.ones(
            size=(input_ids.shape[0], max_length - input_ids.shape[1]),
            dtype=input_ids.dtype) * tokenizer.pad_token_id
        input_ids = torch.cat([input_ids, pad_tube], dim=1)
    
    # 同样处理 labels
    if labels.shape[1] < max_length:
        pad_tube = torch.ones(
            size=(labels.shape[0], max_length - labels.shape[1]),
            dtype=labels.dtype) * IGNORE_INDEX
        labels = torch.cat([labels, pad_tube], dim=1)
    
    # 将 input_ids_system 堆叠成一个 tensor
    input_ids_system = torch.stack(input_ids_system, dim=0)

    # 计算填充后的输入长度，确保它们不会超过 max_length
    # min_max_len = min(
    #     max_length - input_ids_system.shape[-1],
    #     tokenizer.model_max_length - input_ids_system.shape[-1],
    # )

    # input_ids = input_ids[:, :max_length]
    # labels = labels[:, :max_length]

    # 创建最终的 batch 字典
    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),  # attention mask 用于指示哪些部分是填充
        input_ids_system=input_ids_system,
    )

    # 如果每个样本包含图像数据，进行堆叠
    if 'image' in instances[0]:
        images = [instance['image'] for instance in instances]
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['images'] = torch.stack(images)  # 堆叠图像 tensor（bsz, 3, 256, 256）
        else:
            batch['images'] = images  # 如果图像尺寸不一致，则保留为列表

    return batch


def get_personalized_mmu_dataloader(
        data_root,
        concept_name,
        tokenizer,
        batch_size,
        num_workers,
        max_length,
        new_tokens,
        nums_new_token_i: int = 16,
):
    train_dataset = PersonalizedMMUDataset(
        tokenizer,
        data_root,
        concept_name,
        nums_new_token_i,
        new_tokens,
    )
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=partial(
            mmu_collate_fn,
            tokenizer=tokenizer,
            max_length=max_length,
        ),
    )

    return dataloader
    

# class PersonalizedT2IDataset(Dataset):
#     def __init__(
#         self,
#         data_root: str,
#         concept_name: str,
#     ):
#         self.data_root = data_root
#         self.concept_name = concept_name
        
#         case_type = data_w_cases[concept_name]
#         training_img_dir_path = os.path.join(data_root, case_type, "concept/train", concept_name)
#         self.img_paths = []
#         for img in os.listdir(training_img_dir_path):
#             if img.endswith(('png', 'jpg', 'jpeg')) and "mask" not in img:
#                 img_path = os.path.join(training_img_dir_path, img)
#                 self.img_paths.append(img_path)
                
#         assert len(self.img_paths) == 10, f"Expected 10 images for mcllava dataset, found {len(img_paths)}"
        
#     def __len__(self):
#         return len(self.img_paths)
    
#     def __getitem__(self, idx):
#         img_path = self.img_paths[idx]
#         img = Image.open(img_path).convert("RGB")
#         item = {
#             "conditions": f"A photo of <{self.concept_name}>.",
#             "images": img,  # [3, 256, 256] tensor on cpu
#         }
#         item = image_transform_dict(item)
#         return item


class PersonalizedT2IDataset(Dataset):
    def __init__(self, 
                 data_root: str, 
                 concept_name: str, 
                 tokenizer, 
                 max_text_len: int = 128,
                 nums_new_token_i: int = 16
                 ):
        """
        :param data_root: 数据根目录
        :param concept_name: 概念名称，例如 "Alex"
        :param tokenizer: 文本分词器，用于将条件文本转换为 token ids
        :param max_text_len: 条件文本的最大长度（分词后长度）
        """
        self.data_root = data_root
        self.concept_name = concept_name
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        case_type = data_w_cases[concept_name]
        training_img_dir_path = os.path.join(data_root, case_type, "concept/train", concept_name)

        self.img_paths = []
        for img in os.listdir(training_img_dir_path):
            if img.lower().endswith(('png', 'jpg', 'jpeg')) and "mask" not in img:
                img_path = os.path.join(training_img_dir_path, img)
                self.img_paths.append(img_path)
                
        assert len(self.img_paths) == 10, f"Expected 10 images for mcllava dataset, found {len(self.img_paths)}"
        
        # self.system_personalized_prompt = f"<{concept_name}> is "
        # for i in range(nums_new_token_i):
        #     self.system_personalized_prompt += f"<token_{i}>"
        #     if i == nums_new_token_i - 1:
        #         self.system_personalized_prompt += "."
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # 读取图像
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        # 定义条件文本
        condition_text = f"A photo of <{self.concept_name}>."
        # condition_text = "Captain America's shield"
        item = {
            # "conditions": self.system_personalized_prompt + "\n" + condition_text,
            "conditions": condition_text,
            "images": img,  
        }
        # 对图像进行预处理（resize、ToTensor 等）
        item = image_transform_dict(item)
        # # 使用 tokenizer 对条件文本进行编码
        # tokens = self.tokenizer(
        #     condition_text,
        #     max_length=self.max_text_len,
        #     truncation=True,
        #     padding="max_length"
        # )
        # # tokens["input_ids"] 为列表，此处转成 tensor
        # item["input_ids"] = torch.tensor(tokens["input_ids"])
        return item


# def t2i_collate_fn(batch):
    
#     images = torch.stack([item["images"] for item in batch], dim=0)  # (bsz, 3, 256, 256)
#     input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)  # (bsz, max_text_len)
#     return {"images": images, "input_ids": input_ids}


def get_personalized_t2i_dataloader(data_root, concept_name, tokenizer, batch_size, num_workers, max_length=128, nums_new_token_i=16):
    """
    :param data_root: 数据根目录
    :param concept_name: 概念名称
    :param tokenizer: 文本分词器
    :param batch_size: 每个批次样本数
    :param num_workers: DataLoader 使用的进程数
    :param max_text_len: 文本最大长度
    :return: 返回构建好的 DataLoader 对象
    """
    dataset = PersonalizedT2IDataset(data_root, concept_name, tokenizer, max_length, nums_new_token_i)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        # collate_fn=t2i_collate_fn
    )
    return dataloader