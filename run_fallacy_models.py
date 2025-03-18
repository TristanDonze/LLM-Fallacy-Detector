import json
import torch
import argparse
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb 
from datetime import datetime
from peft import PeftModel, PeftConfig

os.environ["HF_HOME"] = "/Data/pbv/huggingface"
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_MAPPING = {
    "phi4": "microsoft/phi-4",
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
    "smollm": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "mistral-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral24b": "mistralai/Mistral-Small-24B-Instruct-2501",
    "llama8b": "meta-llama/Meta-Llama-3-8B",
    "llama3.2": "meta-llama/Llama-3.2-3B",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "qwen": "Qwen/Qwen2.5-14B-Instruct-1M",
}


parser = argparse.ArgumentParser(description="Run a fallacy detection model on a JSON dataset.")
parser.add_argument("--fallacy_model_dir", type=str, required=True, help="Path to the model checkpoint directory.")
parser.add_argument("--input_files", type=str, nargs='+', required=True,
                    help="List of input JSON files to process.")
# parser.add_argument("--output_file", type=str, help="Path to the output JSON file.")
parser.add_argument("--base_model", type=str, default=None, help="adress to base model, such as 'mistralai/Mistral-7B-Instruct-v0.3'")
parser.add_argument("--device", type=str, default="cuda", help="Device to run the model (cpu, cuda, auto).")
parser.add_argument("--max_new_tokens", type=int, default=600, help="Max new token length for model output.")
parser.add_argument("--normal_model", type=str, default=None, help="If specified, use a normal (non-finetuned) model from MODEL_MAPPING.")
parser.add_argument("--test_mode", action='store_true', help="evaluate model on benchmark dataset fallacies_test.json")
# parser.add_argument("--fallback", action="store_true", help="Allow falling back to /Data/pbv if model not found in provided fallacy_model_dir")
args = parser.parse_args()

current_time = datetime.now().strftime("%Y%m%d-%H%M")


# # Ensure correct model directory
checkpoint_path = args.fallacy_model_dir

base_model_name = args.base_model if not args.normal_model else MODEL_MAPPING.get(args.normal_model)
if not base_model_name:
    raise ValueError(f"Model '{args.normal_model}' not found in the mapping.")

tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=os.getenv("HF_HOME"))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if args.normal_model:
    quant_config = BitsAndBytesConfig(load_in_4bit=True)
    model_to_use = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, quantization_config=quant_config, cache_dir=os.getenv("HF_HOME")).to(device)
else:
    quant_config = BitsAndBytesConfig(load_in_4bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, quantization_config=quant_config, cache_dir=os.getenv("HF_HOME")).to(device)
    model_finetuned = PeftModel.from_pretrained(base_model, args.fallacy_model_dir)
    model_to_use = model_finetuned.merge_and_unload().to(device)

# Define the fallacy detector name based on the model path
path_parts = args.fallacy_model_dir.strip("/").split("/")
DETECTOR_NAME = "/".join(path_parts[-2:]) if len(path_parts) >= 2 else path_parts[-1]
DETECTOR_NAME.replace("/", "-")

if args.normal_model:
    DETECTOR_NAME = f"{args.normal_model}_normal"

output_dir = f"./results/new_{current_time}_{DETECTOR_NAME.replace('/', '_')}"

os.makedirs(output_dir, exist_ok=True)

print("checkpoint path:", checkpoint_path)


def detect_fallacies(prompt):
    """
    Runs the model on the given response text to detect logical fallacies.
    Extracts and returns the JSON object from the model's response.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        output = model_to_use.generate(**inputs, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id)

    # Decode output and remove the prompt
    raw_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove the prompt using exact string matching
    cleaned_output = raw_output.replace(prompt.strip(), "").strip()
    # Extract JSON object using regex
    if args.test_mode: #non greedy match (first { to first })
        match = re.search(r"\{.*?\}", cleaned_output, re.DOTALL)
    else: #greedy match (first { to last })
        match = re.search(r"\{.*\}", cleaned_output, re.DOTALL)
    if match:
        try:
            fallacy_json = json.loads(match.group(0))

            # Ensure "explanations" is a dictionary
            if isinstance(fallacy_json.get("explanations"), str):
                fallacy_json["explanations"] = {"note": fallacy_json["explanations"]}

            return fallacy_json
        except json.JSONDecodeError:
            print("Warning: Model output is not valid JSON. Attempting to fix formatting...")

            # Try a manual fix if JSON parsing fails
            fixed_output = match.group(0).replace("'", '"')  # Ensure valid double quotes

            try:
                fallacy_json = json.loads(fixed_output)
                
                # Ensure explanations is a dictionary
                if isinstance(fallacy_json.get("explanations"), str):
                    fallacy_json["explanations"] = {"note": fallacy_json["explanations"]}

                return fallacy_json
            except json.JSONDecodeError:
                print("Critical: Unable to parse JSON. Returning raw text.")
                return {"raw_output": cleaned_output}
    else:
        print("Warning: No JSON detected in model output. Keeping raw text.")
        return {"raw_output": cleaned_output}

if not args.test_mode:

    for input_file in args.input_files:
        input_path = f"./results/results_speech_{input_file}"
        output_path = os.path.join(output_dir, f"analysed_{input_file}") # Save results for current file inside output directory

        # Load dataset
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_items = sum(len(responses) for responses in data.values())
        cur_item = 0

        for question, responses in data.items():
            for response_obj in responses:
                cur_item += 1
                print(f"Processing {cur_item}/{total_items} in {input_file}...")

                response_text = response_obj["response"]
                prompt1 = f"""
    ### Instruction:
    Analyze the following text for logical fallacies. If any fallacies are found, return a structured JSON output like this:

    {{
        "fallacies": ["fallacy_name_1", "fallacy_name_2", ...],
        "explanations": {{
            "fallacy_name_1": "Explanation of the fallacy",
            "fallacy_name_2": "Explanation of the fallacy"
        }}
    }}

    If no fallacies are found, return:

    {{
        "fallacies": ["None"],
        "explanations": {{
            "note": "Explanation of why there is no fallacy"
        }}
    }}

    Text to analyze for potential fallacies:
    {response_text}

    ### Output (JSON format as above):
    """

                # Run fallacy detection
                fallacy_analysis = detect_fallacies(prompt1)
                if cur_item % 10 == 0: # only print 1 tenth of answer
                    print(f"Current model answer:\n {fallacy_analysis}")

                # Add fallacy results to response object
                if "fallacy_detectors" not in response_obj:
                    response_obj["fallacy_detectors"] = {}

                response_obj["fallacy_detectors"][DETECTOR_NAME] = fallacy_analysis

        # Save results for this file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"Updated fallacy analysis saved to {output_path}")


elif args.test_mode: # evaluate on fallacies_test.json
    overall_results = []
    # Expecting exactly one test file passed (e.g., fallacies_test.json)
    input_file = args.input_files[0]
    print(f"Processing test file: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    
    total_count = 0
    correct_count = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for i, test_case in enumerate(test_cases):
        total_count += 1
        example_text = test_case.get("example", "")
        expected_fallacy = test_case.get("fallacy", "N/A").strip().lower()
        print(f"\nAnalyzing example {i+1}:")

        prompt_testing = f"""
    ### Instruction:
    Analyze the following text for a logical fallacy. 

    Here are the allowed fallacy types, from which you must pick one:
    - circular reasoning
    - appeal to anger
    - appeal to pity
    - false dilemma
    - appeal to fear
    - hasty generalization
    - appeal to worse problem
    - appeal to positive emotion
    - fallacy of division
    - no fallacy

    If any fallacy is found, return a structured JSON output like this:

    {{"fallacy": ["fallacy_name"]}}

    If no fallacies are found, return:

    {{"fallacy": ["no fallacy"]}}

    Text to analyze for potential fallacies:
    {example_text}
    ### Output (JSON format as above):
    """
        detected = detect_fallacies(prompt_testing)
        
        # Retrieve the detected fallacy from the model output (assume first element in list)
        detected_fallacy = "no fallacy"
        if "fallacy" in detected and isinstance(detected["fallacy"], list) and len(detected["fallacy"]) > 0:
            detected_fallacy = detected["fallacy"][0].strip().lower()
        
        result = {
            "example": example_text,
            "expected_fallacy": expected_fallacy,
            "detected": detected,
            "detected_fallacy": detected_fallacy
        }
        overall_results.append(result)
        print(f"Expected: {expected_fallacy}")
        print(f"Detected: {detected_fallacy}")
        
        # Count correct predictions
        if detected_fallacy == expected_fallacy:
            correct_count += 1
        
        # True positive: correct fallacy detected
        if detected_fallacy != "no fallacy" and detected_fallacy == expected_fallacy:
            true_positive += 1
        # False positive: wrong fallacy detected
        elif detected_fallacy != "no fallacy" and detected_fallacy != expected_fallacy:
            false_positive += 1
        # True negative: no fallacy detected, expected no fallacy
        elif detected_fallacy == "no fallacy" and expected_fallacy == "no fallacy":
            true_negative += 1
        # False negative: no fallacy detected, but expected fallacy
        elif detected_fallacy == "no fallacy" and expected_fallacy != "no fallacy":
            false_negative += 1

    # Compute metrics
    accuracy = correct_count / total_count if total_count > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Prepare the statistics
    stats_text = f"""
    Evaluation Results:
    -------------------
    Total examples: {total_count}
    Correct predictions: {correct_count}
    Accuracy: {accuracy:.2f}

    Fallacy Detection Metrics (Positive Cases Only):
        True Positives: {true_positive}
        False Positives: {false_positive}
        True Negatives: {true_negative}
        False Negatives: {false_negative}

    Precision: {precision:.2f}
    Recall: {recall:.2f}
    F1 Score: {f1:.2f}

    Additional Observations:
    - Model predicted a fallacy in {true_positive + false_positive} out of {total_count} cases (out of 90 fallacies).
    - Model predicted no fallacy in {true_negative + false_negative} out of {total_count} cases (out of 10 no fallacies).
    """
    print("results are \n", stats_text)
    # Save detailed results to a JSON file
    filename = f"benchmark_{DETECTOR_NAME}_{current_time}"
    filename = filename.replace("/", "_")
    print("filename =", filename)
    filename_json=filename+".json"
    output_file_json = os.path.join("results", filename_json)
    print("output path json is:", output_file_json)
    with open(output_file_json, "w", encoding="utf-8") as out_f:
        json.dump(overall_results, out_f, indent=4, ensure_ascii=False)
    print(f"\nDetailed results saved to {output_file_json}")

    # Save statistics to a text file
    filename_txt=filename+".txt"
    stats_file = os.path.join("results", filename_txt)
    with open(stats_file, "w", encoding="utf-8") as stats_f:
        stats_f.write(stats_text)
    print(f"Statistics saved to {stats_file}")