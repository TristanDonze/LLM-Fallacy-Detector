"""
File for running a debate between two models, for N turns (downloads and loads models if needed first).
Saves conversation in a json file at the end.

Usage:
    Run the script in bash:
    
    python debate_setup.py --models model1,model2 --topic "Debate Topic" --turns N --mode single --trust
    

Arguments:
    --models    (str)  : Comma-separated model names (e.g., "phi4,mistral").
                         The first model is assigned as model1, the second as model2.
    --topic     (str)  : The topic of the debate.
    --turns     (int)  : Number of turns in the debate (each turn is one model speaking).
    --mode      (str)  : "single" (default) for local execution, "distributed" for multi-GPU/machine execution.
    --trust     (flag) : Enables trust_remote_code=True when downloading models (optional).

"""



####### imports
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import datetime
import torch.cuda
import subprocess
#######

# # Huggingface login to avoid api/token errors:
# from huggingface_hub import login
# login(token="huggingface_token")  


MODEL_MAPPING = {
    "phi4": "microsoft/phi-4",
    "smollm": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "llama8B": "meta-llama/Meta-Llama-3-8B",
    "llama3.2": "meta-llama/Llama-3.2-3B",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
}


model_path = "/Data/pbv"
os.makedirs(model_path, exist_ok=True)


# Define the remote servers to run one model on each
SERVER1 = "tarse"
SERVER2 = "phalange"


# SETUP 
#### Download models, tokenizers, and store them in /Data/pbv/ (Polytechnique PCs)
def download_and_save_models(model_names, save_dir="/Data/pbv/", trust=False):
    os.makedirs(save_dir, exist_ok=True)
    
    for model_name in model_names:
        hf_model_name = MODEL_MAPPING.get(model_name, model_name)
        model_dir = os.path.join(save_dir, model_name)
        if os.path.exists(model_dir):
            print(f"Model '{model_name}' already exists in '{model_dir}'. Skipping download.")
            continue
        print(f"Downloading model: {model_name} ({hf_model_name})")

        try:
            model = AutoModelForCausalLM.from_pretrained(hf_model_name, cache_dir=save_dir, trust_remote_code=trust)
            tokenizer = AutoTokenizer.from_pretrained(hf_model_name, cache_dir=save_dir, trust_remote_code=trust)
            os.makedirs(model_dir, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            print(f"Model '{model_name}' saved to '{model_dir}'.")
        except Exception as e:
            print(f"Error downloading model '{model_name}': {e}")


def load_models(model1_name, model2_name, base_dir="/Data/pbv/"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model1_path = os.path.join(base_dir, model1_name)
    model2_path = os.path.join(base_dir, model2_name)
    print(f"Loading model 1: {model1_name} from {model1_path}")
    model1 = AutoModelForCausalLM.from_pretrained(model1_path).to(device)
    print(f"{model1_name} loaded")
    tokenizer1 = AutoTokenizer.from_pretrained(model1_path)
    print(f"Tokenizer 1 loaded")
    print(f"Loading model 2: {model2_name} from {model2_path}")
    model2 = AutoModelForCausalLM.from_pretrained(model2_path).to(device)
    print(f"{model2_name} loaded")
    tokenizer2 = AutoTokenizer.from_pretrained(model2_path)
    print(f"Tokenizer 2 loaded")
    
    return model1, tokenizer1, model2, tokenizer2

###################################################################################


# running on 1 pc only (two small models)
# def debate_turn(model, tokenizer, prompt, max_new_tokens=100, device="cuda"):
#     """Runs a single turn of the debate."""
#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
#     attention_mask = (input_ids != tokenizer.pad_token_id).to(device)
#     output_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=True)
#     response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     print("###################### \n prompt used:", prompt, "\n response given:", response)
#     return response.strip()

# def run_debate(model1, tokenizer1, model2, tokenizer2, topic, turns=6):
#     """Runs a debate between two models and stores their prompts and responses."""
#     conversation = []

#     # Initial prompts for both models
#     model1_prompt = f"You are debating on the topic: {topic}. Present your argument."
#     model2_prompt = f"Your opponent has started the debate on: {topic}. Here is their statement:\n"

#     # Model1 starts
#     response1 = debate_turn(model1, tokenizer1, model1_prompt)
#     conversation.append({"speaker": "Model1", "prompt": model1_prompt, "response": response1})

#     # Model2 responds with full context of Model1's response
#     response2 = debate_turn(model2, tokenizer2, f"Opponent's argument: {response1}\n\nYour response:")
#     conversation.append({"speaker": "Model2", "prompt": f"Opponent's argument: {response1}\n\nYour response:", "response": response2})

#     # Debate loop, ensuring each model gets full context
#     for turn in range(turns - 2):
#         current_model, current_tokenizer = (model1, tokenizer1) if turn % 2 == 0 else (model2, tokenizer2)
#         speaker = "Model1" if turn % 2 == 0 else "Model2"

#         # Build the prompt with all previous statements
#         prompt = f"{speaker}, present your next statement:\n" + "\n".join(
#             [f"{c['speaker']}: {c['response']}" for c in conversation]
#         )

#         # Generate the next response and add it to the conversation
#         response = debate_turn(current_model, current_tokenizer, prompt)
#         conversation.append({"speaker": speaker, "prompt": prompt, "response": response})

#     return conversation



def debate_turn(model, tokenizer, prompt, max_new_tokens=100, device="cuda"):
    """Runs a single turn of the debate."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).to(device)  # Create attention mask
    output_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=True)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("######################################## \n prompt used:", prompt, "\n response given:", response)
    return response.strip()

def run_debate(model1, model1_name, tokenizer1, model2, model2_name, tokenizer2, topic, turns=6):
    """Runs a debate between two models and stores their prompts and responses."""
    conversation = []

    # Initial prompts for both models
    model1_prompt = f"You are a debating AI. You will now debate on the topic: {topic}. Present your argument."
    model2_prompt = f"You are a debating AI. Your opponent has started the debate on: {topic}. Here is their statement:\n"

    # Model1 starts
    response1 = debate_turn(model1, tokenizer1, model1_prompt)
    conversation.append({"speaker": model1_name, "prompt": model1_prompt, "response": response1})

    # Model2 responds with full context of Model1's response
    model2_prompt_with_response = f"Opponent's argument: {response1}\n\nYour response:"
    response2 = debate_turn(model2, tokenizer2, model2_prompt_with_response)
    conversation.append({"speaker": model2_name, "prompt": model2_prompt_with_response, "response": response2})

    # Debate loop, ensuring each model gets full context
    for turn in range(turns - 2):
        current_model, current_tokenizer = (model1, tokenizer1) if turn % 2 == 0 else (model2, tokenizer2)
        speaker = model1_name if turn % 2 == 0 else model2_name

        # Build the prompt with all previous statements
        prompt = f"{speaker}, present your next statement:\n" + "\n".join(
            [f"{c['speaker']}: {c['response']}" for c in conversation]
        )

        # Generate the next response and add it to the conversation
        response = debate_turn(current_model, current_tokenizer, prompt)
        conversation.append({"speaker": speaker, "prompt": prompt, "response": response})

    return conversation








#################### Run on two separate PCs
def run_model_remotely(server, model_name, prompt):
    remote_command = (
        f"ssh {server} 'python3 -c \"\"\""
        "import torch\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "model_path = \"/Data/pbv/{model_name}\"\n"
        "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n"
        "model = AutoModelForCausalLM.from_pretrained(model_path).to(device)\n"
        "tokenizer = AutoTokenizer.from_pretrained(model_path)\n"
        "input_ids = tokenizer.encode(\"{prompt}\", return_tensors=\"pt\").to(device)\n"
        "output_ids = model.generate(input_ids, max_new_tokens=100, do_sample=True)\n"
        "generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)\n"
        "print(generated_text)\"\"\"'"
    )
    result = subprocess.run(remote_command, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def dist_run_debate(model1_name, model2_name, topic, turns=6):
    conversation = []
    model1_intro = f"""You are debating on: {topic}. Please state your position."""
    model2_intro = f"""Your opponent has started the debate on: {topic}. Here is their statement:\n"""

    conversation.append({"speaker": "model1", "text": model1_intro})
    response1 = run_model_remotely(SERVER1, model1_name, model1_intro)
    conversation.append({"speaker": "model1", "text": response1})

    model2_prompt = model2_intro + f"Model1: {response1}\nModel2: "
    response2 = run_model_remotely(SERVER2, model2_name, model2_prompt)
    conversation.append({"speaker": "model2", "text": response2})

    for turn in range(turns - 2):
        speaker = "model1" if turn % 2 == 0 else "model2"
        server = SERVER1 if speaker == "model1" else SERVER2
        model_name = model1_name if speaker == "model1" else model2_name
        prompt = f"Debate Topic: {topic}\n" + "\n".join([f"{c['speaker']}: {c['text']}" for c in conversation]) + f"\n{speaker}:"
        response = run_model_remotely(server, model_name, prompt)
        conversation.append({"speaker": speaker, "text": response.strip()})
    
    return conversation
#################################


######## Results file
def save_conversation(conversation, model_name, mode, topic, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    cleaned_topic = topic.replace(" ", "-").replace(",", "").replace(":", "").replace("?", "")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(results_dir, f"debate_{cleaned_topic}_{model_name}_{mode}_{timestamp}.json")
    cleaned_filename = os.path.join(results_dir, f"cleaned_debate_{cleaned_topic}_{model_name}_{mode}_{timestamp}.json")

    # Save full conversation with prompts and responses
    with open(filename, "w") as f:
        json.dump(conversation, f, indent=2)

    # Save cleaned conversation with only responses
    cleaned_conversation = [{"speaker": entry["speaker"], "response": entry["response"]} for entry in conversation]
    with open(cleaned_filename, "w") as f:
        json.dump(cleaned_conversation, f, indent=2)

    print(f"Debate saved to {filename}")
    print(f"Cleaned debate saved to {cleaned_filename}")




#### Main part and arg passing
import argparse
def main():
    parser = argparse.ArgumentParser(description="AI Debate Framework")
    parser.add_argument("--models", type=str, default="phi4,phi4", help="Comma-separated model names (e.g., 'phi4,mistral') The list of models we use is defined at the top of the file. First model is model1")
    parser.add_argument("--topic", type=str, default="The impact of AI on society",help="Debate topic.")
    parser.add_argument("--turns", type=int, default=6, help="Number of debate turns (each turn is one model speaking)")
    parser.add_argument("--mode", type=str, choices=["single", "distributed"], default="single",help="Run on a single machine or distribute across multiple GPUs/machines.")
    parser.add_argument("--trust", action="store_true", help="Enable trust_remote_code=True for downloading models")

    args = parser.parse_args()

    model_names = [name.strip() for name in args.models.split(",")]
    if len(model_names) < 2:
        print("Error: Please provide at least two model names separated by a comma.")
        return

    if args.mode == "single":
        download_and_save_models(model_names, trust=args.trust)
        model1, tokenizer1, model2, tokenizer2 = load_models(model_names[0], model_names[1])
        model1name, model2name = model_names[0], model_names[1]
        print(f"starting debate with {model2name} versus {model2name}")
        conversation = run_debate(model1, model1name, tokenizer1, model2, model2name, tokenizer2, args.topic, args.turns)

        save_conversation(conversation, model_names[0], args.mode, args.topic)

    else:
        # Distributed mode logic.
        # For example, you could call your dispatch functions similar to the distributed_master.py.
        # This block can later use SSH and remote execution to run debate turns on multiple servers.
        print("Distributed mode is selected. Implement remote dispatch logic here using SSH and your job scheduler.")
        # Example: dispatch_debate_job(topic=args.topic, turns=args.turns, models=model_names, ...)

if __name__ == "__main__":
    main()




# Function to run model inference remotely via SSH
# def run_model_remotely(server, model_name, prompt):
#     remote_command = (
#         f"ssh {server} 'python3 -c """
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# model_path = \"/Data/pbv/{model_name}\"
# device = \"cuda\" if torch.cuda.is_available() else \"cpu\"
# model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# input_ids = tokenizer.encode(\"{prompt}\", return_tensors=\"pt\").to(device)
# output_ids = model.generate(input_ids, max_new_tokens=100, do_sample=True)
# generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
# print(generated_text)""'"
#     )
#     result = subprocess.run(remote_command, shell=True, capture_output=True, text=True)
#     return result.stdout.strip()


# def run_debate(model1_name, model2_name, topic, turns=6):
#     conversation = []
#     model1_intro = f"""You are debating on: {topic}. Please state your position."""
#     model2_intro = f"""Your opponent has started the debate on: {topic}. Here is their statement:\n"""

#     conversation.append({"speaker": "model1", "text": model1_intro})
#     response1 = run_model_remotely(SERVER1, model1_name, model1_intro)
#     conversation.append({"speaker": "model1", "text": response1})

#     model2_prompt = model2_intro + f"Model1: {response1}\nModel2: "
#     response2 = run_model_remotely(SERVER2, model2_name, model2_prompt)
#     conversation.append({"speaker": "model2", "text": response2})

#     for turn in range(turns - 2):
#         speaker = "model1" if turn % 2 == 0 else "model2"
#         server = SERVER1 if speaker == "model1" else SERVER2
#         model_name = model1_name if speaker == "model1" else model2_name
#         prompt = f"Debate Topic: {topic}\n" + "\n".join([f"{c['speaker']}: {c['text']}" for c in conversation]) + f"\n{speaker}:"
#         response = run_model_remotely(server, model_name, prompt)
#         conversation.append({"speaker": speaker, "text": response.strip()})
    
#     return conversation

# def main():
#     parser = argparse.ArgumentParser(description="Distributed AI Debate")
#     parser.add_argument("--models", type=str, default="phi4,mistral", help="Comma-separated model names")
#     parser.add_argument("--topic", type=str, default="The impact of AI on society", help="Debate topic")
#     parser.add_argument("--turns", type=int, default=6, help="Number of debate turns")
#     args = parser.parse_args()

#     model_names = args.models.split(",")
#     if len(model_names) < 2:
#         print("Error: Provide two model names separated by a comma.")
#         return

#     conversation = run_debate(model_names[0], model_names[1], args.topic, args.turns)
#     save_conversation(conversation)