"""
File for running a speech, or a debate between two models for N turns (downloads and loads models if needed first).
Can also have a debate between human and AI.
Saves conversation in a json file at the end.

Usage:
    To run file in terminal:
    
    python debate_main.py --models model1,model2 --topic "Debate Topic" --turns num_turns --mode single --topics_file filename 

    examples: 
    - to run speeches by a given model
        python debate_main.py --models mistral-instruct --quant 4bit --speech_mode --topics_file ./data/1000_topics.json --response_length medium
    
    - to run a speech by trump
        python debate_main.py --models NOT_USED_IN_TRUMP_MODE_BUT_REQUIRED --speech_mode --topics_file ./data/1000_topics.json --trump_mode
    
    - to run a debate between a human and phi4
        python debate_main.py --models human,phi4 --topic "Gun control in the USA" --turns 6 --mode single
    

Arguments:
--models : the model names to use. ex: "phi4-mini, llama3.2" or "phi4" if --speech_mode is set. 
--quant : the required quantization in ["32bit", "16bit", "8bit", "4bit"].
--speech_mode : if written, sets speech_mode to True, and so now requires 1 model, for which a speech will be required. 
    If not written, the default debate mode is used between two models.
--topic : a topic on which the debate (or speech will take place). Used for running debate/speech on a single topic.
--topics_file : a json file in which the values are topics. Used for running debate/speech on all these topics.
--topics_list : a list filled with topics. Used for running debate/speech on all these topics.
--N : the number of topics to take from the topics_file (or topics_list), we take only the first N. If not set, all will be used. 
--turns : the number of turns for the debate (ignored if --speech_mode is used). 1 turn = 1 model speaking.
--response_length : takes three values, "short", "medium", or "long". Has an impact on length of responses from models.
--mode : Useless for now since distributed not implemented yet. Sets whether to run on a single machine (--mode single) or on several machines (--mode distributed. )
 

For reference, here are the available models:
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

"""

import argparse
from manage_models import ModelManager
from debate_setup import Debater
import time
import json
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
 
def main():

    parser = argparse.ArgumentParser(description="AI Debate Framework")
    # parameters for model type and quantization
    parser.add_argument("--models", type=str, default="phi4,phi4",
                        help="Comma-separated model names. Use 'human' for human input.")
    parser.add_argument("--quant", type=str, choices=["32bit", "16bit", "8bit", "4bit"], default=None,
                    help="Quantization level (default: None). If it fails, will fall back to lower.")

    parser.add_argument("--speech_mode", action="store_true", default=False,
                    help="If set, run in speech mode (one speech only by 1 model) instead of debate mode.")

    # parameters for debate format
    parser.add_argument("--topic", type=str, default="The impact of AI on society",
                        help="Debate topic.")
    parser.add_argument("--turns", type=int, default=6,
                        help="Number of debate turns (each turn is one model speaking)")
    parser.add_argument("--response_length", type=str, choices=["short", "medium", "long"], default="medium",
                    help="Sets response length of models.")

    parser.add_argument("--mode", type=str, choices=["single", "distributed"], default="single",
                        help="Run on a single machine or distribute across multiple GPUs/machines.")
    
    # parameters for looped execution (generating results)
    parser.add_argument("--topics_file", type=str, help="Path to a file containing topics (one per line).")
    parser.add_argument("--topics_list", type=str, help="Comma-separated list of topics.")
    parser.add_argument("--N", type=int, default=None, help="Number of topics to run (from the start of the list).")

    parser.add_argument("--trump_mode", action="store_true", default=False,
                    help="If set, run trump inference from the trump finetuned model")


    args = parser.parse_args()

    model_names = [name.strip() for name in args.models.split(",")]

    manager = ModelManager(trust_remote_code=True)

    if not args.trump_mode:
        manager.download_and_save(model_names)

    results = {}

    if args.speech_mode:
        if len(model_names) != 1:
            logger.error("Error: In speech mode, only one model should be provided. Example usage: --models mistral --speech_mode")
            return

        if args.trump_mode: #trump_mode bypasses all logic (naturally)
            model, tokenizer = manager.load_model("placeholder_not_used_blablabla", trump_mode=True)
        else: #load normally (no trump)
            model, tokenizer = manager.load_model(model_names[0], num_models=1, quantization=args.quant)
        if model is None:
            logger.error("Failed to load model, exiting.")
            return

        # Read topics
        if args.topics_file:
            logger.info(f"Using the topics_file: {args.topics_file}")
            with open(args.topics_file, 'r', encoding='utf-8') as f:
                topics_dict = json.load(f)
                topics = list(topics_dict.values())
        elif args.topics_list:
            logger.info(f"Using the topics_list: {args.topics_list}")
            topics = [t.strip() for t in args.topics.split(",")]
        else:
            logger.info(f"Using the single topic: {args.topic}")
            topics = [args.topic]

        if args.N is not None:
            topics = topics[:args.N]
        total = len(topics)
        for index, topic in enumerate(topics):
            logging.info(f"[{index+1}/{total}] Current topic being processed: {topic}.")
            debate_runner = Debater(
                manager, 
                model_names[0],
                None,
                topic,
                response_length=args.response_length,
                turns=1,
                specified_quant=args.quant,
                speech_mode=True,
                model1=model,
                tokenizer1=tokenizer,
                trump_mode=args.trump_mode
            )
            conversation = debate_runner.run_speech()
            # debate_runner.save_conversation(conversation, args.mode, args.speech_mode, args.response_length)
            results[topic] = conversation
            # save results
        if args.trump_mode:
            results_filename = f"results_speeches_trump_finetuned.json"
        else: 
            results_filename = f"results_speech_{model_names[0]}_{args.quant}_{args.N if args.N else 'all'}_{args.response_length}.json"
        with open(f"results/{results_filename}", 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"All conversations saved to results/{results_filename}")

    else: # normal debate mode requires two models
        if len(model_names) < 2:
            logger.error("Error: Please provide two model names (or one being 'human') for a debate.")
            return

        if args.mode == "single":
            # manager = ModelManager(trust_remote_code=True)
            # manager.download_and_save(model_names)
            debate_runner = Debater(
                    manager, 
                    model_names[0],
                    model_names[1], 
                    args.topic, 
                    speech_mode=False,
                    response_length=args.response_length, 
                    turns=args.turns, 
                    specified_quant=args.quant
                )
            conversation = debate_runner.run_debate()

            # debate_runner.save_conversation(conversation, args.mode, args.speech_mode, args.response_length)

            results[args.topic] = conversation
            # results_filename = f"results_debate_{model_names[0], model_names[1]}_{args.quant}_{args.N if args.N else 'all'}_{args.response_length}.json"
            # with open(f"results/{results_filename}", 'w') as f:
            #     json.dump(results, f, indent=4)
            # logger.info(f"All conversations saved to results/{results_filename}")

        elif args.mode == "distributed":
            logger.info("Distributed across two pcs (not implemented yet)")
    

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f"Total execution took {round(end_time-start_time, 1)} seconds.")
