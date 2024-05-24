from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


path_to_adapter = 'qwen_output'

model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="auto",
).eval()



merged_model = model.merge_and_unload()
# max_shard_size and safe serialization are not necessary. 
# They respectively work for sharding checkpoint and save the model to safetensors
merged_model.save_pretrained('merge_lora', max_shard_size="2048MB", safe_serialization=True)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    path_to_adapter, # path to the output directory
)
tokenizer.save_pretrained('merge_lora')
