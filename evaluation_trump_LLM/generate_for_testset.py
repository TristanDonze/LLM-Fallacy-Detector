import torch
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_path = "./model"
checkpoint_path = "./checkpoint_trump_QLORA"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16 
)

device_map = "auto" if torch.cuda.is_available() else None

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(
    base_model,
    checkpoint_path,
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def get_testset():
    dataset_path  = "All_Speeches_Instruction.jsonl"
    dataset = load_dataset("json", data_files=dataset_path)
    dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
    test_dataset = dataset["test"]
    return test_dataset

def tokenize_sample(sample):
    instruction = sample["instruction"]
    output = sample["output"]
    input_instruction = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True)
    target = tokenizer(output, return_tensors="pt", padding=True, truncation=True)
    len_output = len(target["input_ids"][0])
    return input_instruction, len_output


import json

def write_to_jsonl(data_dict):
    with open("generated_takes.jsonl", "a") as jsonl_file:
        jsonl_file.write(json.dumps(data_dict) + "\n")
        
        
def generate_trump_take(sample):
    input_instruction, len_output = tokenize_sample(sample)
    input_ids = input_instruction["input_ids"].to(model.device)
    attention_mask = input_instruction["attention_mask"].to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=len_output,
            temperature=0.7,
            top_p=0.3,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=True,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

import os

progress_file = "progress.txt"
if os.path.exists(progress_file):
    with open(progress_file, "r") as f:
        n_done = int(f.read().strip())
else:
    n_done = 0

testset = get_testset()
testset_length = len(testset)

for i, sample in enumerate(testset):
    if i < n_done:
        continue
    if i in to_skip:
        continue
    try:
        print(f"Done: {i + 1} / {testset_length}")
        output = generate_trump_take(sample)
        result = {'instruction': sample["instruction"], 'output_golden': sample["output"], 'output_generated': output}
        write_to_jsonl(result)

        n_done = i + 1
        with open(progress_file, "w") as f:
            f.write(str(n_done))
    except ValueError as e:
        print(f"Skipping sample {i} due to ValueError: {e}")
        continue