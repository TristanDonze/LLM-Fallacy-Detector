import os
import torch
import gc  
import time
import logging
import shutil

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

MODEL_MAPPING = {
    "phi4": "microsoft/phi-4",
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
    "phi4-instruct": "microsoft/Phi-4-multimodal-instruct",
    "smollm": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "mistral-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama8b": "meta-llama/Meta-Llama-3-8B",
    "llama3.2": "meta-llama/Llama-3.2-3B",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
}



class ModelManager:
    def __init__(self, base_dir="/Data/pbv/", trust_remote_code=True):
        self.base_dir = base_dir
        self.trust = trust_remote_code
        os.makedirs(self.base_dir, exist_ok=True)

    def download_and_save(self, model_names):
        for model_name in model_names:
            if model_name.lower() == "human":
                logger.info("That model is a pretrained human, the downloading is already done :)")
                continue  # if human, no need to load a model
            hf_model_name = MODEL_MAPPING.get(model_name, model_name)
            model_dir = os.path.join(self.base_dir, model_name)
            print(f"model dir is {model_dir}")

            if os.path.exists(model_dir):
                logger.info(f"Model '{model_name}' already exists at {model_dir}. Skipping download.")
                continue

            logger.info(f"Downloading model: {model_name} ({hf_model_name})")
            try:
                model = AutoModelForCausalLM.from_pretrained(hf_model_name, cache_dir=self.base_dir, trust_remote_code=self.trust)
                tokenizer = AutoTokenizer.from_pretrained(hf_model_name, cache_dir=self.base_dir, trust_remote_code=self.trust)
                os.makedirs(model_dir, exist_ok=True)
                model.save_pretrained(model_dir)
                tokenizer.save_pretrained(model_dir)
                logger.info(f"Model '{model_name}' saved to '{model_dir}'.")
            except Exception as e:
                logger.error(f"Error downloading model '{model_name}': {e}")


    def load_model(self, model_name, num_models=2,quantization=None):
        """
        Loads model in required quantization if specified, otherwise loads in largest quantization that fits in memory.
        If num_models is 2, attempts to load both models ensuring they fit within half of available GPU memory each.
        """
        if model_name.lower() == "human":
            logger.info("That model is a pretrained human, the loading is already done :)")
            return "human", None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_dir = os.path.join(self.base_dir, model_name)
        logger.info(f"Loading model '{model_name}' from {model_dir}")

        quant_options = {
            "32bit": None,
            "16bit": {"torch_dtype": torch.float16},
            "8bit": BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True),
            "4bit": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        }

        quantization_order = [quantization] if quantization else ["16bit", "8bit", "4bit"]

        available_memory_per_model = get_available_memory_per_model(num_models)
        model = None #init model variable

        for quant in quantization_order:
            logger.info(f"Trying to load model '{model_name}' with {quant} quantization.")
            try:
                clean_up_memory(model)

                logger.info(f"Memory used BEFORE loading '{model_name}' with {quant} quantization:")
                print_memory_usage()

                model = load_with_quantization(model_dir, quant, device, quant_options)
                
                logger.info(f"Memory used after loading '{model_name}' with {quant} quantization:")
                print_memory_usage()

                if num_models == 2: # limited to half the memory for each model
                    used_memory = torch.cuda.memory_allocated() / 1024**2
                    if used_memory > available_memory_per_model:
                        logger.warning(f"Model '{model_name}' with {quant} quantization exceeds available memory per model ({used_memory:.2f} MB > {available_memory_per_model:.2f} MB). Trying lower quantization...\n\n")
                        clean_up_memory(model)
                        model = None
                        continue 

                tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=self.trust)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token  # Fix missing pad_token
                return model, tokenizer

            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    logger.error(f"Out of memory error while loading '{model_name}' with {quant} quantization. Trying lower quantization...\n\n")
                    clean_up_memory(model)
                    model = None
                    continue
                else:
                    logger.error(f"Failed to load '{model_name}' with {quant} quantization: {e}\n\n")
                    clean_up_memory(model)
                    model = None
                    continue

            except Exception as e:
                logger.error(f"Error loading model '{model_name}' with {quant} quantization: {e}\n\n")
                clean_up_memory(model)
                model = None
                continue

        logger.error(f"Error: Unable to load '{model_name}' even with 4-bit quantization.")
        return None, None


    def generate_response(self, model, tokenizer, prompt, response_length):
        length_modes = {
            "short": 250,
            "medium": 500,
            "long": 1000
        }
        max_tokens = length_modes.get(response_length, 250) #default to "medium" length

        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).to(device)
        output = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_new_tokens=max_tokens, 
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            return_legacy_cache=True,
            # output_scores=True,
            # num_beams=5,
            # early_stopping = True
        )
        output_ids = output.sequences
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return response.strip()




########## Helper functions for ModelManager class ######################""

def print_memory_usage():
    """ Helper function to print memory usage details """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    else:
        logger.info("CUDA is not available, skipping memory reporting.")

def clean_up_memory(to_remove):
    if to_remove is not None:
        del to_remove
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)
    torch.cuda.reset_peak_memory_stats()

def load_with_quantization(model_dir, quant, device, quant_options=None):
    """ Load model with specific quantization and device options """
    if quant == "32bit":
        model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    elif quant == "16bit":
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="cuda"
        )
    elif quant == "8bit":
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=quant_options[quant],
            device_map="cuda"
        )
    elif quant == "4bit":
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=quant_options[quant],
            device_map="cuda"
        )

    return model


def get_available_memory_per_model(num_models=1):
    free_memory = torch.cuda.mem_get_info()[0] / 1024**2
    logger.info(f"Available GPU memory: {free_memory:.2f} MB")
    return free_memory / num_models