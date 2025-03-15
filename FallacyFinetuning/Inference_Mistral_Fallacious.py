import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_path = "./model"
checkpoint_path = "./checkpoint_fallacies"

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

model.eval()

def generate_response(
    prompt, 
    model, 
    tokenizer, 
    max_length=300, 
    temperature=0.9,
    top_p=0.6,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3
):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=True,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

text = "Speaker A : I therefore wish to gradually phase out nuclear energy as renewable energy develops.\nSpeaker B : You want to dismantle the nuclear industry?"
prompt = f"### Analyze the following text for logical fallacies:\n\n {text}\n\n### Response:\n"
response = generate_response(prompt, model, tokenizer)
print(response)