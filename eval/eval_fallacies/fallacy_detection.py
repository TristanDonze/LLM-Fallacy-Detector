import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define model directory (Change if using another checkpoint)
MODEL_DIR = "./FallacyModel/checkpoint_fallacious/checkpoint-5/"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto")

# Load input JSONL file
INPUT_FILE = "fallacy_analysis.json"
OUTPUT_FILE = "fallacy_analysis_detected.json"

# Load the dataset
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Function to generate fallacy detection response
def detect_fallacies(response_text):
    """
    Runs the model on the given response text to detect logical fallacies.
    Returns the model's output.
    """
    prompt = f"""
    ### Instruction:
    Analyze the following text for logical fallacies: {response_text}

    ### Output: 
    """
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=4096, pad_token_id=tokenizer.eos_token_id)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Define the fallacy detector name
DETECTOR_NAME = "FallacyModel-checkpoint-5"

# Process each question-response pair
for question, responses in data.items():
    for response_obj in responses:
        response_text = response_obj["response"]

        # Run fallacy detection
        fallacy_analysis = detect_fallacies(response_text)

        # Add fallacy results to the response object
        if "fallacy_detectors" not in response_obj:
            response_obj["fallacy_detectors"] = {}

        response_obj["fallacy_detectors"][DETECTOR_NAME] = {
            "fallacies_found": fallacy_analysis,  # Model-generated analysis
            "explanations": fallacy_analysis  # Could refine this later if different
        }

# Save the updated JSON file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Updated fallacy analysis saved to {OUTPUT_FILE}")
