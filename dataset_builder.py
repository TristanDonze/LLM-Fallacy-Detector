"""
File for creating datasets for testing our results.
Downloaded the file from: https://github.com/pillowsofwind/DebateQA/blob/main/dataset/test.jsonl
We extract the questions
"""

import json

def extract_topics(dataset_path):
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

# Example usage
dataset_path = 'test.jsonl'
extract_topics(dataset_path)
