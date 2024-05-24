from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import transformers
from transformers import Trainer, GPTQConfig, deepspeed, BitsAndBytesConfig, DataCollatorForSeq2Seq
import json
import os
import torch 
import pandas  as pd 
from datasets import Dataset

# from supervised_dataset import LazySupervisedDataset, SupervisedDataset
from model_save import safe_save_model_for_hf_trainer


model_name_or_path = '/data/liucd/BigModel/Qwen1.5-1.8B-Chat'
data_path = 'huanhuan.json'

@dataclass
class ModelArguments:
    model_name_or_path: str = model_name_or_path


@dataclass
class DataArguments:
    data_path: str = field(default=data_path, metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",  "up_proj", "gate_proj","down_proj",]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    use_lora: bool = True
    bf16: bool = False
    output_dir: str = 'qwen_output'
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )  # 微调时最大序列长度
    gradient_checkpointing: bool = True
    report_to: str = 'none'
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 8   # bs=1对应的训练集loss 更低 !
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_beta2: float = 0.95
    warmup_ratio: float = 0.01
    lr_scheduler_type: str = 'cosine'
    logging_steps: int = 1  # 每隔10个打印一次日志

    # deepspeed: str = '/data/liucd/BigModel/qwen/Qwen/finetune/ds_config_zero2.json'


args_model = ModelArguments()
args_train = TrainingArguments()
args_lora = LoraArguments()
args_data = DataArguments()


# tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    args_model.model_name_or_path,
    model_max_length=args_train.model_max_length,
    padding_side="right",
    use_fast=False,
)

compute_dtype = torch.bfloat16 if args_train.bf16  else torch.float32


# Load model and tokenizer
config = transformers.AutoConfig.from_pretrained(
    args_model.model_name_or_path,
)

model = transformers.AutoModelForCausalLM.from_pretrained(
        args_model.model_name_or_path,
        config=config,
        device_map='auto',
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        if args_train.use_lora and args_lora.q_lora
        else None,
        torch_dtype=compute_dtype  # add by liucd 否则在4卡上会float32运行
    )


print(model.dtype)


lora_config = LoraConfig(
            r=args_lora.lora_r,
            lora_alpha=args_lora.lora_alpha,
            target_modules=args_lora.lora_target_modules,
            lora_dropout=args_lora.lora_dropout,
            bias=args_lora.lora_bias,
            task_type="CAUSAL_LM",
        )


if args_lora.q_lora:
     model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=args_train.gradient_checkpointing
            )  # 将某些的LN层等从FP16变成FP32


model = get_peft_model(model, peft_config=lora_config)
model.print_trainable_parameters()

# 调用 model.enable_input_require_grads() 是为了确保在使用 grad_checkpoint 时，模型的输入能够被要求梯度，以便在检查点处能够正确地重新计>算梯度。
if args_train.gradient_checkpointing:
    model.enable_input_require_grads()



# 将JSON文件转换为CSV文件
df = pd.read_json('./weather.json')
ds = Dataset.from_pandas(df)

# sys = '现在你要扮演皇帝身边的女人--甄嬛'
sys = '你是一个人工智能助手'
def process_func(example):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|im_start|>system\n {sys} <|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
tokenized_id

tokenizer.decode(tokenized_id[2]['input_ids'])

trainer = Trainer(
    model=model,
    args=args_train,
    train_dataset=tokenized_id,
    tokenizer=tokenizer,
    # train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

trainer.save_state()  # 保存状态

safe_save_model_for_hf_trainer(trainer=trainer, output_dir=args_train.output_dir, bias=args_lora.lora_bias)

