from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import transformers
from transformers import Trainer, GPTQConfig, deepspeed
import json
import os
import torch


model = transformers.AutoModelForCausalLM.from_pretrained(
        # '/data/liucd/Qwen-1_8B-Chat',
        '/data/liucd/BigModel/Qwen1.5-1.8B-Chat',
        device_map='auto',    # 分布式需要注释掉或者设置为None
        
        torch_dtype=torch.float16
)


print(model.dtype)
