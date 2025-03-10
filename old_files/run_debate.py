"""
File for running a debate between two models, for N turns (downloads and loads models if needed first).
Can also have a debate between human and AI
Saves conversation in a json file at the end.

Usage:
    To run file in terminal:
    
    python run_debate.py --models model1,model2 --topic "Debate Topic" --turns N --mode single

    example: 
    python run_debate.py --models human,phi4 --topic "Gun control in the USA" --turns 6 --mode single
    

Arguments:
    --model(s)    (str)  : Comma-separated model names (e.g., "phi4,mistral").
                         The first model is assigned as model1, the second as model2.
                         If "human" is included, then will wait for human inputs to the debate.
    --topic     (str)  : The topic of the debate.
    --turns     (int)  : Number of turns in the debate (each turn is one model speaking).
    --mode      (str)  : "single" (default) for local execution, "distributed" for multi-GPU/machine execution.

"""

import argparse
from manage_models import ModelManager
from debate_setup import Debater

def main():
    parser = argparse.ArgumentParser(description="AI Debate Framework")
    parser.add_argument("--models", type=str, default="phi4,phi4",
                        help="Comma-separated model names. Use 'human' for human input.")
    parser.add_argument("--quant", type=str, choices=["32bit", "16bit", "8bit", "4bit"], default=None,
                    help="Quantization level (default: 32bit). If it fails, will fall back to lower.")

    parser.add_argument("--topic", type=str, default="The impact of AI on society",
                        help="Debate topic.")
    parser.add_argument("--turns", type=int, default=6,
                        help="Number of debate turns (each turn is one model speaking)")
    parser.add_argument("--response_length", type=str, choices=["short", "medium", "long", "4bit"], default="medium",
                    help="Sets response length of models.")

    parser.add_argument("--mode", type=str, choices=["single", "distributed"], default="single",
                        help="Run on a single machine or distribute across multiple GPUs/machines.")

    args = parser.parse_args()

    model_names = [name.strip() for name in args.models.split(",")]
    if len(model_names) < 2:
        print("Error: Please provide two model names (or one of them being 'human') separated by a comma as model1,human or model1,model2.")
        return

    if args.mode == "single":
        manager = ModelManager(trust_remote_code=True)
        manager.download_and_save(model_names)

        
        print(f"Setting up debate for {model_names[0]} against {model_names[1]}")
        if "human" in model_names:
            num_models_to_load = 1
        else: num_models_to_load = 2
        print(f"loading {num_models_to_load} models currently:")
        
        # Load models with requested quantization
        model1, tokenizer1 = manager.load_model(model_names[0], args.quant, num_models=num_models_to_load)
        print(f"loaded model1: {model_names[0]}")
        model2, tokenizer2 = manager.load_model(model_names[1], args.quant, num_models=num_models_to_load)
        print(f"loaded model2: {model_names[1]}")

        if model1 is None or model2 is None:
            print("Error: Failed to load one or both models.")
            return

        # create a debater instance and pass the loaded models to it
        debate_runner = Debater(manager, model_names[0], model_names[1], args.topic, args.turns)
        debate_runner.model1, debate_runner.tokenizer1 = model1, tokenizer1
        debate_runner.model2, debate_runner.tokenizer2 = model2, tokenizer2

        conversation = debate_runner.run_debate(args.response_length)
        debate_runner.save_conversation(conversation, args.mode)

    elif args.mode == "distributed":
        print("Distributed across two pcs (not implemented yet)")

if __name__ == "__main__":
    main()
