"""
This file is used for testing a single model on a single prompt. 
We mainly used this file to check if a model "fits" in the computer's memory without crashing (to fix some of our bugs).

For reference, here are the available models:
MODEL_MAPPING = {
    "phi4": "microsoft/phi-4",
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
    "smollm": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "llama8b": "meta-llama/Meta-Llama-3-8B",
    "llama3.2": "meta-llama/Llama-3.2-3B",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
}
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from manage_models import ModelManager

manager = ModelManager(trust_remote_code=True) # sets up model from 

# Choose the model key from the MODEL_MAPPING to test
model_key = "llama8b"  # pick model to test (only use the key in the dictionary above)

quantization = None  # "32bit", "16bit", "8bit", or "4bit"

model, tokenizer = manager.load_model(model_key, quantization=quantization, num_models=1)

if model and tokenizer:
    print(f"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA Successfully loaded model: {model_key} with {quantization if quantization else 'default'} quantization")

    print("the bos_token_id is:", tokenizer.bos_token_id)

    prompt = """
        Answer the following question.
        What is the capital of France? Tell me 3 things about this city.
        Once you have answered this question, you MUST end your response with "END_OF_RESPONSE".
    """
    try:
        response = manager.generate_response(model, tokenizer, prompt, max_new_tokens=1000)
        print("BBBBBBBBBBBBBBBBBBB Response:", response)
    except Exception as e:
        print(f"CCCCCCCCCCCCCCCCCCCCCCC Error generating response: {e}")
else:
    print(f"DDDDDDDDDDDDDDDDDDD Failed to load model: {model_key}")
