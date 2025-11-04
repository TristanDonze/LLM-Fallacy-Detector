import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from manage_models import ModelManager
from peft import PeftModel, PeftConfig

device = "cuda" if torch.cuda.is_available() else "cpu"


# Available models
MODEL_MAPPING = {
    "phi4": "microsoft/phi-4",
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
    "phi4-instruct": "microsoft/Phi-4-multimodal-instruct",
    "smollm": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "mistral-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral24b": "mistralai/Mistral-Small-24B-Instruct-2501",
    "llama8b": "meta-llama/Meta-Llama-3-8B",
    "llama3.2": "meta-llama/Llama-3.2-3B",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "qwen": "Qwen/Qwen2.5-14B-Instruct-1M",
}

# Argument parser for command-line options
parser = argparse.ArgumentParser(description="Run a model inference test.")
parser.add_argument("--model_key", type=str, help="Key of the model to load from MODEL_MAPPING.")
parser.add_argument("--quantization", type=str, choices=["32bit", "16bit", "8bit", "4bit", None], default=None, help="Quantization mode.")
parser.add_argument("--normal", type=str, choices=["32bit", "16bit", "8bit", "4bit", None], default=None, help="Quantization mode.")
args = parser.parse_args()

# Path to local model checkpoint (if no model_key is provided)
model_dir = './FallacyModel/checkpoint_fallacious/checkpoint-5/'

# Load model using ModelManager or from a local directory
if args.model_key:
    if args.model_key not in MODEL_MAPPING:
        raise ValueError(f"Invalid model key. Choose from: {list(MODEL_MAPPING.keys())}")
    
    print(f"Loading model: {args.model_key} from Hugging Face...")
    manager = ModelManager(trust_remote_code=True)
    model, tokenizer = manager.load_model(args.model_key, quantization=args.quantization, num_models=1)

else:
    print(f"Loading local model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)



    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, cache_dir=os.getenv("HF_HOME")).to(device)
    model_finetuned = PeftModel.from_pretrained(base_model, args.fallacy_model_dir)
    model_to_use = model_finetuned.merge_and_unload().to(device)


# Ensure model is loaded
if not model or not tokenizer:
    print("Failed to load model!")
    exit(1)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the prompt
# Are you tired of being ignored by your government?  Is it right that the top 1% have so much when the rest of us have so little?  I urge you to vote for me today!'
prompt = """
####
Tell me what the capital of France is.
"""

# Tokenize input and move tensors to the correct device
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate response
outputs = model.generate(
    inputs["input_ids"], 
    attention_mask=inputs["attention_mask"],
    max_length=300, 
    num_return_sequences=1, 
    no_repeat_ngram_size=2, 
    pad_token_id=tokenizer.eos_token_id
)

# Decode output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated response: {generated_text}")


#################
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Path to the local model checkpoint
# model_dir = '/Data/pbv/mistral24b'

# # Load model and tokenizer from the local directory
# tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
# model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)

# # Ensure model and tokenizer are loaded
# if not model or not tokenizer:
#     print("Failed to load model or tokenizer!")
#     exit(1)

# # Move model to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Define the prompt
# prompt = """
# ### Instruction:
# Analyze the following text for logical fallacies: 'I like turtles.'

# ### Output: 
# """

# # Tokenize input and move tensors to the correct device
# inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
# inputs = {k: v.to(device) for k, v in inputs.items()}

# # Generate response
# outputs = model.generate(
#     inputs["input_ids"], 
#     attention_mask=inputs["attention_mask"],
#     max_length=300, 
#     num_return_sequences=1, 
#     no_repeat_ngram_size=2, 
#     pad_token_id=tokenizer.eos_token_id
# )

# # Decode output
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(f"Generated response: {generated_text}")
