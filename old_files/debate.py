import os
import json
import datetime

class DebateRunner:
    def __init__(self, model_manager, model1_name, model2_name, topic, turns=6):
        self.model_manager = model_manager
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.topic = topic
        self.turns = turns
        self.conversation = []

        # Load models (or set to None if human)
        self.model1, self.tokenizer1 = self.model_manager.load_model(model1_name)
        self.model2, self.tokenizer2 = self.model_manager.load_model(model2_name)

    def debate_turn(self, speaker, prompt, model, tokenizer):
        if speaker.lower() == "human": # Get human input (human vs AI debate)
            response = input(f"{prompt}\nYour response: ")
            return response.strip()
        else:
            response = self.model_manager.generate_response(model, tokenizer, prompt)
            print("########################################")
            print("Prompt used:", prompt)
            print("Response given:", response)
            return response

    def run_debate(self):
        # Initial prompts for both speakers.
        model1_prompt = f"You are a debating AI. Debate on the topic: {self.topic}. Present your argument."
        model2_prompt = f"Your opponent started the debate on: {self.topic}. Here is their statement:\n"

        # First turn: model1 (or human) starts.
        response1 = self.debate_turn(self.model1_name, model1_prompt, self.model1, self.tokenizer1)
        self.conversation.append({"speaker": self.model1_name, "prompt": model1_prompt, "response": response1})
        
        # Second turn: model2 (or human) responds.
        prompt2 = f"Opponent's argument: {response1}\n\nYour response:"
        response2 = self.debate_turn(self.model2_name, prompt2, self.model2, self.tokenizer2)
        self.conversation.append({"speaker": self.model2_name, "prompt": prompt2, "response": response2})
        
        # Continue the debate loop.
        for turn in range(self.turns - 2):
            if turn % 2 == 0:
                speaker = self.model1_name
                model, tokenizer = self.model1, self.tokenizer1
            else:
                speaker = self.model2_name
                model, tokenizer = self.model2, self.tokenizer2

            prompt = f"{speaker}, present your next statement:\n" + "\n".join(
                [f"{entry['speaker']}: {entry['response']}" for entry in self.conversation]
            )
            response = self.debate_turn(speaker, prompt, model, tokenizer)
            self.conversation.append({"speaker": speaker, "prompt": prompt, "response": response})
        
        return self.conversation

    def save_conversation(self, conversation, mode):
        # Clean the topic string for filenames.
        cleaned_topic = self.topic.replace(" ", "-").replace(",", "").replace(":", "").replace("?", "")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        filename = os.path.join(results_dir, f"debate_{cleaned_topic}_{self.model1_name}_{mode}_{timestamp}.json")
        cleaned_filename = os.path.join(results_dir, f"cleaned_debate_{cleaned_topic}_{self.model1_name}_{mode}_{timestamp}.json")

        # Save full conversation with prompts.
        with open(filename, "w") as f:
            json.dump(conversation, f, indent=2)
        # Save a cleaned version with only responses.
        cleaned_conversation = [{"speaker": entry["speaker"], "response": entry["response"]} for entry in conversation]
        with open(cleaned_filename, "w") as f:
            json.dump(cleaned_conversation, f, indent=2)

        print(f"Debate saved to {filename}")
        print(f"Cleaned debate saved to {cleaned_filename}")
