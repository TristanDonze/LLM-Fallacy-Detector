"""
File for creating datasets

"""
import json
import os
# from langchain_ollama import OllamaLLM
import pandas as pd
import json
import os
import shutil

# Dataset of questions/debate topics #####


def extract_topics(dataset_path):
     # Downloaded the file from: https://github.com/pillowsofwind/DebateQA/blob/main/dataset/test.jsonl
    topics = []
    topics_counter = 0
    with open(dataset_path, 'r') as file:
        for line in file:
            topics_counter += 1
            data = json.loads(line)
            topics.append(data["question"])

    topics_dict = {i + 1: topic for i, topic in enumerate(topics)}

    with open(f"{topics_counter}_topics.json", 'w') as file:
        json.dump(topics_dict, file, indent=4)

# dataset_path = '/data/topics.jsonl'
# extract_topics(dataset_path)


###################################


# Dataset for Fallacy finetuning

def convert_dataset(input_file, output_file):
    # used dataset: https://github.com/ChadiHelwe/MAFALDA/blob/main/datasets/gold_standard_dataset.jsonl
    converted_data = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            
            # Extract sentence-level fallacies
            sentence_labels = json.loads(data["sentences_with_labels"])
            comments = data.get("comments", ["No explanation provided."])

            for sentence, label_groups in sentence_labels.items():
                sentence = sentence.strip().replace("\n", " ")  # Clean sentence formatting

                # Flatten nested lists and filter out "nothing"
                fallacies = set(f for labels in label_groups for f in labels if f != "nothing")

                if not fallacies:
                    output_text = "No fallacy detected."
                else:
                    # Match comments to fallacies (generalized)
                    relevant_explanations = [
                        comment for comment in comments 
                        if any(fallacy in comment.lower() for fallacy in fallacies)
                    ]
                    explanation = " ".join(relevant_explanations) if relevant_explanations else "No explanation provided."

                    output_text = f"{', '.join(fallacies)}: {explanation}"

                # Create instruction-output pair
                entry = {
                    "instruction": f"Identify any logical fallacies in the following statement:\n'{sentence}'",
                    "output": output_text
                }

                converted_data.append(entry)

    # Save as JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for item in converted_data:
            f.write(json.dumps(item) + "\n")

    print("Dataset successfully converted.")

# Get the directory where the script is located
here = os.path.dirname(os.path.abspath(__file__))

# Construct full paths to the input and output files
input_path = os.path.join(here, "FallacyModel", "gold_standard_dataset.jsonl")
output_path = os.path.join(here, "FallacyModel", "fallacy_instructions.jsonl")

# Call the function with the correct paths
# convert_dataset(input_path, output_path)


def generate_finetune_entry(input_data):
    # used dataset: https://huggingface.co/datasets/tasksource/logical-fallacy/viewer/default/train?p=26&views%5B%5D=train&row=2611
    """Generates a single instruction-output JSON object from the full text."""
    
    text = input_data["text"].strip().replace("\n", " ")
    labels = input_data["labels"]
    comments = input_data.get("comments", [])

    # Extract fallacies (excluding "nothing")
    fallacies = sorted(set(label[2] for label in labels if label[2] != "nothing"))

    if not fallacies:
        fallacy_text = "No fallacies."
        explanation_text = ""
    else:
        fallacy_text = f"This text contains {', '.join(fallacies)}."
        explanation_text = " ".join(comments) if comments else "No explanation provided."

    output_text = f"{fallacy_text} {explanation_text}".strip()

    # Create instruction-output pair
    instruction_data = {
        "instruction": f"Analyze the following text for logical fallacies: '{text}'",
        "output": output_text
    }

    return instruction_data


# Define dataset splits
splits = {
    "train": "data/train-00000-of-00001-8c3d4e48fe0f561b.parquet",
    "test": "data/test-00000-of-00001-ce92752fd4455cd1.parquet",
    "dev": "data/dev-00000-of-00001-99b3373cde156b17.parquet"
}

# Load dataset from Hugging Face storage
train_df = pd.read_parquet("hf://datasets/tasksource/logical-fallacy/" + splits["dev"])

# Function to create instruction-output pairs
def generate_instruction_output_pairs(df, output_file):
    instruction_output_pairs = []

    for _, row in df.iterrows():
        text = row["source_article"]  # Extract text
        fallacy = row["logical_fallacies"]  # Extract fallacy label

        # Create instruction-output pair
        instruction = f"Analyze the following text for logical fallacies:\n\n{text}"
        output = f"This text contains {fallacy}."

        instruction_output_pairs.append({
            "instruction": instruction,
            "output": output
        })

    # Save as JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for pair in instruction_output_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Output saved to '{output_file}'.")

# Run for train split
# output_file = "dev_instruction_output.jsonl"
# generate_instruction_output_pairs(train_df, output_file)


input_files = [
    "train_instruction_output.jsonl",
    "test_instruction_output.jsonl",
    "dev_instruction_output.jsonl",
    "fallacy_instructions.jsonl"
]

output_file = "fallacy_full_dataset.jsonl"

# Concatenate files
with open(f"FallacyModel/{output_file}", "wb") as outfile:
    for file in input_files:
        with open(f"FallacyModel/{file}", "rb") as infile:
            shutil.copyfileobj(infile, outfile)

print(f"All datasets merged into '{output_file}'.")


