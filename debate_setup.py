import os
import json
import datetime
import logging
import re

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


# Templates for structured prompting
prompt_templates = {
    "first_speaker": {
        "short": "You are [MODEL]. Your stance is fixed and unwavering. State your argument concisely, defending your position without deviation. \n RULES: Your speech should not exceed 3 sentences. Your speech MUST be formatted as:\n[BEG_OF_ANS] Your speech here [END_OF_ANS]\n",
        "medium": "You are [MODEL]. Your position is clear and unchanging. Present a quick, structured argument with solid reasoning, concrete examples, and key counterpoints that support your stance. \n RULES: Your speech should not exceed 6 sentences. Your speech MUST be formatted as:\n[BEG_OF_ANS] Your speech here [END_OF_ANS]\n",
        "long": "You are [MODEL]. Remain resolute and never waver. Provide a detailed argument including historical context, relevant statistics, and robust counterpoints that demonstrate why your position is the only valid perspective. \n RULES: Your speech should not exceed 15 sentences. Your speech MUST be formatted as:\n[BEG_OF_ANS] Your speech here [END_OF_ANS]\n"
    },
    "responder": {
        "short": "You are [MODEL]. Briefly counter your opponent's argument while firmly maintaining your own fixed stance. \n RULES: Your speech should not exceed 3 sentences. Your speech MUST start with [BEG_OF_ANS] and end with [END_OF_ANS].\n",
        "medium": "You are [MODEL]. Provide a concise and focused rebuttal to your opponent's statement, using clear reasoning and examples to defend your consistent position. \n RULES: Your speech should not exceed 6 sentences. Your speech MUST start with [BEG_OF_ANS] and end with [END_OF_ANS].\n",
        "long": "You are [MODEL]. Deliver a detailed rebuttal with logic, evidence, and alternative viewpoints—always reaffirming your unwavering stance. \n RULES: Your speech should not exceed 15 sentences. Your speech MUST start with [BEG_OF_ANS] and end with [END_OF_ANS].\n"
    },
    
    
    "speech": {
    "short": """Defend your position without deviation, very concisely, in the following format:

[BEG_OF_ANS] {Your argument here.} [END_OF_ANS]

Do not provide backstory or any additional context. Your speech must end with [END_OF_ANS].
        """,

    "medium": """Defend your position without deviation. Present a structured argument with solid reasoning and concrete examples that support your stance, in the following format:

[BEG_OF_ANS] {Your argument here.} [END_OF_ANS]
        
Do not provide backstory or any additional context. Your speech must end with [END_OF_ANS].
        """,

    "long": """Defend your position without deviation. Present a detailed and structured argument with historical context, statistics and examples that support your stance, in the following format::

[BEG_OF_ANS] {Your argument here.} [END_OF_ANS]
        
Do not provide backstory or any additional context. Your speech must end with [END_OF_ANS].
        """
    }

}

class Debater:
    """ 
    Class handling the debate:
        - calls the model manager to download, save, and then load the models into GPU memory
        - sets up the turn taking logic for the debate (either between two models, or a model and a human)
        - saves the final debate into a json file
    """
    def __init__(self, model_manager, model1_name, model2_name, topic, response_length, turns=6, specified_quant=None, speech_mode=False, model1=None, tokenizer1=None, model2=None, tokenizer2=None):
        self.model_manager = model_manager
        self.model1_name = model1_name
        self.model2_name = model2_name if not speech_mode else None
        self.topic = topic
        self.turns = 1 if speech_mode else turns
        self.conversation = []
        self.response_length = response_length
        self.speech_mode = speech_mode

        logger.info(f"Setting up {'speech' if speech_mode else 'debate'} for {model1_name}")

        num_models_to_load = 1 if self.speech_mode or "human" in [model1_name, model2_name] else 2
        # logger.info(f"loading {num_models_to_load} model(s) currently:")

        #Load models
        # self.model1, self.tokenizer1 = self.model_manager.load_model(model1_name, num_models=num_models_to_load, quantization=specified_quant)
        # logger.info(f"Finished loading {model1_name}.")

        # if not self.speech_mode: # only load second model if not in speech_mode
        #     self.model2, self.tokenizer2 = self.model_manager.load_model(model2_name, num_models=num_models_to_load, quantization=specified_quant)
        #     logger.info(f"Finished loading {model2_name}.")
        if model1 is None:
            self.model1, self.tokenizer1 = self.model_manager.load_model(model1_name, num_models=num_models_to_load, quantization=specified_quant)
            logger.info(f"Finished loading {model1_name}.")
        else:
            logger.info("Model already loaded")
            self.model1 = model1
            self.tokenizer1 = tokenizer1

        if not self.speech_mode:
            if model2 is None:
                self.model2, self.tokenizer2 = self.model_manager.load_model(model2_name, num_models=num_models_to_load, quantization=specified_quant)
                logger.info(f"Finished loading {model2_name}.")
            else:
                logger.info("Model already loaded")
                self.model2 = model2
                self.tokenizer2 = tokenizer2
            

        if self.model1 is None or (not self.speech_mode and self.model2 is None):
            logger.error("Failed to load one or both models")
            return

    ########## helper functions #######################
    def get_human_input(self, prompt):
        response = input(f"{prompt}\nYour response: ")
        return response.strip()

    def get_ai_response(self, model, tokenizer, prompt):
        response = self.model_manager.generate_response(model, tokenizer, prompt, response_length=self.response_length)
        # print("#########################\nActual prompt:", prompt)
        # print("#########################\nRaw model response:", response)

        response = response.replace(prompt, "").strip()
        # print("#########################\nAfter stripping prompt:", response)

        beg_marker = "[BEG_OF_ANS]"
        end_marker = "[END_OF_ANS]"

        if beg_marker in response and end_marker in response: # both markers, keep in between
            answer = response.split(beg_marker)[1].split(end_marker)[0].strip()
            # print("#########################\nExtracted answer between markers:", answer)
        elif end_marker in response: # only end marker, keep whats before
            answer = response.split(end_marker)[0].strip()
            # print("#########################\nExtracted answer before end marker:", answer)
        elif beg_marker in response: # only beginning marker, keep whats after
            answer = response.split(beg_marker)[1].strip()
            # print("#########################\nExtracted answer after beginning marker:", answer)
        else:
            # No markers found, return the entire response
            answer = response
            # print("#########################\nNo markers found, returning full response:", answer)

        return answer


    def check_if_debate_over(self, response, model_name):
        if "/bye" in response.strip().lower():
            logger.info(f"{model_name} ended the debate early (/bye in response).")
            return True
        else: return False

    def build_full_prompt(self, speaker):
        """Builds a concise and structured debate prompt without redundant instructions."""
        
        conversation_str = f"\nThe topic of the debate is: {self.topic}.\n\n"
        for entry in self.conversation:
            conversation_str += f"{entry['speaker']}: {entry['response']}\n\n"

        prompt_type = "first_speaker" if len(self.conversation) == 0 else "responder"
        structured_prompt = prompt_templates[prompt_type][self.response_length].replace("[MODEL]", speaker)

        full_prompt = f"{conversation_str}{speaker}: {structured_prompt}"

        return full_prompt.strip()


    ##############################################

    def debate_turn(self, speaker, model, tokenizer):
        """
        Handles each turn of the debate.
        AI models receive full debate history; humans get a prompt.
        """
        if speaker.lower() == "human":
            return self.get_human_input(self.build_full_prompt(speaker))


        full_debate_prompt = self.build_full_prompt(speaker)
        response = self.get_ai_response(model, tokenizer, full_debate_prompt)
        return response

    def run_debate(self):

        logger.info(f"Debate started on topic: {self.topic}, with maximum response length (for models) of {self.response_length}.")
        logger.info(f"Participants: {self.model1_name} vs {self.model2_name}")

        # Initial prompt
        model1_prompt = "You are {model}, a debating AI with a fixed stance. Debate on the topic: {topic}. {prompt}".format(
            model=self.model1_name,
            topic=self.topic,
            prompt=prompt_templates['first_speaker'][self.response_length].replace("[MODEL]", self.model1_name)
        )

        # First turn: model1 (or human) starts.
        response1 = self.debate_turn(self.model1_name, self.model1, self.tokenizer1)
        self.conversation.append({"speaker": self.model1_name, "prompt": model1_prompt, "response": response1})
        if self.check_if_debate_over(response1, self.model1_name): return self.conversation
        
        # Second turn: model2 (or human) responds.
        response2 = self.debate_turn(self.model2_name, self.model2, self.tokenizer2)
        self.conversation.append({"speaker": self.model2_name, "response": response2})
        if self.check_if_debate_over(response2, self.model2_name): return self.conversation
        
        # Continue the debate loop.
        for turn in range(self.turns - 2):
            if turn % 2 == 0:
                speaker = self.model1_name
                model, tokenizer = self.model1, self.tokenizer1
            else:
                speaker = self.model2_name
                model, tokenizer = self.model2, self.tokenizer2

            response = self.debate_turn(speaker, model, tokenizer)

            self.conversation.append({"speaker": speaker, "response": response})
            if self.check_if_debate_over(response, speaker): return self.conversation
        
        return self.conversation
    
    def run_speech(self):
        """Run a single-turn speech where the model delivers one answer to the prompt."""
        prompt = "Explain your stance on: {topic}. {prompt_template}".format(
            model=self.model1_name,
            topic=self.topic,
            prompt_template=prompt_templates["speech"][self.response_length].replace("[MODEL]", self.model1_name)
        )
        response = self.get_ai_response(self.model1, self.tokenizer1, prompt)
        self.conversation.append({"speaker": self.model1_name, "prompt": prompt, "response": response})
        return self.conversation


    def save_conversation(self, conversation, mode, speech_mode, length):
        cleaned_topic = self.topic.replace(" ", "-").replace(",", "").replace(":", "").replace("?", "")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        filename = os.path.join(results_dir, f"{'speech' if speech_mode else 'debate'}_{self.model1_name}_{length}_{cleaned_topic}_{mode}_{timestamp}.json")
        cleaned_filename = os.path.join(results_dir, f"cleaned_{'speech' if speech_mode else 'debate'}_{self.model1_name}_{length}_{cleaned_topic}_{mode}_{timestamp}.json")

        # Save full conversation with prompts.
        with open(filename, "w") as f:
            json.dump(conversation, f, indent=2)
        logger.info(f"Debate saved to {filename}")

        # Save a cleaned version with only responses.
        # cleaned_conversation = [{"speaker": entry["speaker"], "response": entry["response"]} for entry in conversation]
        # with open(cleaned_filename, "w") as f:
        #     json.dump(cleaned_conversation, f, indent=2)
        # logger.info(f"Cleaned debate saved to {cleaned_filename}")



