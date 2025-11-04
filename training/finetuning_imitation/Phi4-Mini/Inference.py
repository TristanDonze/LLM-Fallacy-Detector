import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

checkpoint_path = "./checkpoint_trump"

base_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-4-mini-instruct", torch_dtype=torch.bfloat16)

model = PeftModel.from_pretrained(base_model, checkpoint_path)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, model, tokenizer, max_length=100, temperature = 0.6):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            max_length=max_length,
            temperature=temperature, # diversity 
            top_p=0.7,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3, 
            do_sample=True,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

subject = "woke culture and why it's a threat to America"
prompt = f"###Instruction:\nImitate Donald Trump's speech style on {subject}\n### Response:\n"
response = generate_response(prompt, model, tokenizer)
print(response)
