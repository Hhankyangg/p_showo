import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image

from models import Showo, MAGVITv2
from training.prompting_utils import UniversalPrompting, create_attention_mask_for_mmu, create_attention_mask_for_mmu_vit
from training.utils import get_config, flatten_omega_conf, image_transform
from transformers import AutoTokenizer
from models.clip_encoder import CLIPVisionTower
from transformers import CLIPImageProcessor
from llava.llava import conversation as conversation_lib
from omegaconf import DictConfig, ListConfig, OmegaConf

conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
config = OmegaConf.load('configs/showo_demo_w_clip_vit.yaml')
# device setup
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side ="left")
uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

# setting up the visual question answering model: magvit-v2
vq_model = MAGVITv2
vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
vq_model.requires_grad_(False)
vq_model.eval()

# setting up vision tower: clip-vit
vision_tower_name ="openai/clip-vit-large-patch14-336"
vision_tower = CLIPVisionTower(vision_tower_name).to(device)
clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)

# setting up the showo model 
model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(device)
# model.eval()

# setting up the parameters
temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 1  # retain only the top_k most likely tokens, clamp others to have 0 probability
SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. " \
                "The assistant gives helpful, detailed, and polite answers to the user's questions."
SYSTEM_PROMPT_LEN = 28