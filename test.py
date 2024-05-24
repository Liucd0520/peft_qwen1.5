from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from model_save import safe_save_model_for_hf_trainer

model_path = '/data/liucd/BigModel/Qwen1.5-1.8B-Chat'
# 将JSON文件转换为CSV文件
df = pd.read_json('./huanhuan.json')
ds = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer

def process_func(example):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|im_start|>system\n现在你要扮演皇帝身边的女人--甄嬛<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
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

import torch

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.float16)
model

model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法

from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=64, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
config

model = get_peft_model(model, config)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir="./output/Qwen1.52",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=1,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to = 'none'
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()
trainer.save_state()  # 保存状态

safe_save_model_for_hf_trainer(trainer=trainer, output_dir=args.output_dir) #, bias=args_lora.lora_bias)
