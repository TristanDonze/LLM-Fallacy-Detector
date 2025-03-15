import os
import sys
import csv
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
from tqdm import tqdm

import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
from huggingface_hub import snapshot_download

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRITON_CACHE_DIR"] = ".triton/cache"

class EmptyCacheCallback(TrainerCallback):
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Optionnel : on peut choisir de le faire moins fréquemment, par exemple tous les 10 steps
        if state.global_step % 10 == 0:
            import torch
            torch.cuda.empty_cache()
        return control


class MetricsLoggingCallback(TrainerCallback):
    def __init__(self, file_path="metrics_log_fallacious.csv"):
        self.file_path = file_path
        with open(file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["step", "metric", "value"])

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            with open(self.file_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                for metric, value in logs.items():
                    writer.writerow([step, metric, value])
        return control

@dataclass
class FineTuneConfig:
    """
    Holds the main hyperparameters and file paths.
    """
    model_name_or_path: str = field(
        default="./model",
        metadata={"help": "Base model checkpoint (locally downloaded) to fine-tune."}
    )
    dataset_path: str = field(
        default="instructions.jsonl",
        metadata={"help": "Path to the JSONL dataset with instruction/output pairs."}
    )
    output_dir: str = field(
        default="./checkpoint_fallacies",
        metadata={"help": "Directory to store final model and checkpoints."}
    )
    # Training hyperparameters
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=2)
    learning_rate: float = field(default=5e-6)
    logging_steps: int = field(default=1)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    seed: int = field(default=42)
    # LoRA hyperparameters
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)
    # Mixed precision
    bf16: bool = field(default=True)
    # Resume from checkpoint
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a checkpoint folder to resume training."}
    )


class FineTuner:
    """Class to handle dataset loading, training, and evaluation with LoRA."""
    def __init__(self, config: FineTuneConfig):
        self.config = config

        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.logger.info("Initializing FineTuner with config:")
        self.logger.info(self.config)

        # Prepare LoRA config
        self.peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["k_proj","q_proj","v_proj", "o_proj"]
        )

        # Prepare training args
        self.train_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            logging_first_step=True,
            save_steps=self.config.save_steps,
            eval_strategy="steps",   # Evaluate every save_steps
            eval_steps=self.config.eval_steps,
            save_total_limit=1,
            bf16=self.config.bf16,
            seed=self.config.seed,
            load_best_model_at_end=False,  # Could set True if using a metric to compare
            gradient_checkpointing=True,
            gradient_accumulation_steps=8,
            # If you want to resume automatically:
            # resume_from_checkpoint=self.config.resume_from_checkpoint,
        )
        
        self.logger.info(f"DeepSpeed Trainer Batch Params:")
        self.logger.info(f"    per_device_train_batch_size: {self.config.per_device_train_batch_size}")
        self.logger.info(f"    gradient_accumulation_steps: {self.train_args.gradient_accumulation_steps}")
        self.logger.info(f"    Calculated train_batch_size: {self.config.per_device_train_batch_size * self.train_args.gradient_accumulation_steps * 1}")


    def load_dataset(self):
        """Loads the JSONL dataset and splits 90-10 into train/test."""
        self.logger.info(f"Loading dataset from {self.config.dataset_path}")
        dataset = load_dataset("json", data_files=self.config.dataset_path)

        self.logger.info("Splitting dataset 90-10 for train/test.")
        dataset = dataset['train'].train_test_split(test_size=0.1, seed=self.config.seed)
        self.train_dataset = dataset["train"]
        self.test_dataset = dataset["test"]

    def preprocess_function(self, example):
        """
        Converts each record {"instruction": "...", "output": "..."}
        into a single text field with an instruction/response template.
        """
        instruction = example["instruction"]
        output = example["output"]

        # Simple formatting
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        return {"text": text}

    def prepare_data(self):
        """Applies the preprocessing with a progress bar."""
        self.logger.info("Applying preprocessing to training set...")
        self.train_dataset = self.train_dataset.map(
            self.preprocess_function,
            remove_columns=self.train_dataset.column_names,
            desc="Preprocessing train dataset",
        )
        self.logger.info("Applying preprocessing to test set...")
        self.test_dataset = self.test_dataset.map(
            self.preprocess_function,
            remove_columns=self.test_dataset.column_names,
            desc="Preprocessing test dataset",
        )

    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer with QLoRA in 4 bits."""
        self.logger.info(f"Loading model from {self.config.model_name_or_path}...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        model_kwargs = {
            "trust_remote_code": True,
            "use_cache": False,
            "torch_dtype": torch.bfloat16 if self.config.bf16 else torch.float32,
            "quantization_config": bnb_config,
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            **model_kwargs
        )

        self.model = prepare_model_for_kbit_training(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        if self.tokenizer.pad_token is None:
            # If no pad_token, just assign eos_token as pad
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.tokenizer.padding_side = "right"
        self.tokenizer.model_max_length = 4096
        self.logger.info("Model and tokenizer loaded with QLoRA 4-bit settings.")

    def create_trainer(self):
        """Create the SFTTrainer with LoRA config."""
        self.logger.info("Creating the SFTTrainer with LoRA config.")
        self.trainer = SFTTrainer(
            model=self.model,
            args=self.train_args,
            peft_config=self.peft_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            processing_class=self.tokenizer,
            callbacks=[MetricsLoggingCallback("metrics_log_fallacious.csv"), EmptyCacheCallback()]
        )

    def train(self):
        """Run the training, with the possibility to resume from checkpoint."""
        resume_checkpoint = self.config.resume_from_checkpoint
        self.logger.info(f"Starting training. Resume? {resume_checkpoint}")
        train_result = self.trainer.train(resume_from_checkpoint=resume_checkpoint)
        metrics = train_result.metrics

        self.logger.info("Training complete. Metrics:")
        self.logger.info(metrics)

        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()
        self.trainer.save_model(self.config.output_dir)

    def evaluate(self):
        """
        Evaluate on the test set. We'll calculate perplexity from the eval_loss
        if it's provided by trainer.evaluate().
        """
        self.logger.info("Evaluating on the test set...")
        metrics = self.trainer.evaluate()
        if "eval_loss" in metrics:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        self.logger.info(metrics)

        self.trainer.save_metrics("eval", metrics)


def main():
    config = FineTuneConfig(
        model_name_or_path="./model",  # dossier local où se trouve le modèle
        dataset_path="fallacy_full_dataset.jsonl",
        output_dir="./checkpoint_fallacies",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        learning_rate=5e-4,
        logging_steps=1,
        save_steps=5,
        seed=42,
        # Si vous avez un checkpoint, vous pouvez le préciser ici :
        # resume_from_checkpoint="./checkpoint_fallacies"
    )

    # Build the FineTuner
    finetuner = FineTuner(config)
    finetuner.load_dataset()
    finetuner.prepare_data()
    finetuner.load_model_and_tokenizer()
    finetuner.create_trainer()

    # Train and Evaluate
    finetuner.train()
    finetuner.evaluate()


if __name__ == "__main__":
    main()
