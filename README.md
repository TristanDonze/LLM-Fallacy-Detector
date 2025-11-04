# Fine-tuning LLMs to Mimic Political Speeches and Detect Fallacious Arguments

**Authors:** Tristan Donzé, Pablo Bertaud-Velten
**Institution:** Institut Polytechnique de Paris

---

## Overview

This project explores the dual challenge of generating political speeches in Donald Trump's rhetorical style and detecting logical fallacies in political discourse. We fine-tuned several LLMs using QLoRA to create both a fallacy detection system and a model that imitates Donald Trump's speech style.

---

## Key Features

- **Trump-Style Speech Generation:** A model fine-tuned to mimic Donald Trump's rhetorical style.
- **Fallacy Detection:** A system to detect logical fallacies in political arguments.
- **Quantitative Analysis:** Evaluation of fallacious arguments in generated speeches compared to baseline models.

---

## Datasets

### Fallacy Detection Datasets
- **Sources:**
  - [MAFALDA benchmark](https://github.com/ChadiHelwe/MAFALDA)
  - [CausalNLP dataset](https://aclanthology.org/2022.findings-emnlp.532/)
  - [HuggingFace dataset](https://huggingface.co/datasets/MrOvkill/fallacies-fallacy-base/viewer/default/train?row=7&views%5B%5D=train)
- **Total:** ~5700 annotated samples
- **Preprocessing:** Formatted as instruction-output pairs for fallacy detection.


### Trump Speeches Datasets
- **Sources:**
  - [Kaggle Trump Speech Dataset](https://www.kaggle.com/datasets/tangtaidje/donald-trump-rev-com-speech-transcripts-dataset)
  - [GitHub Trump Speeches Repository](https://github.com/PedramNavid/trump_speeches)
  - [Roll Call Website](https://rollcall.com/factbase/trump/transcripts/)
- **Total:** 630 speeches
- **Preprocessing:** Instruction-output pairs generated using Gemini's API.


---

## Methodology

### Models Fine-tuned
- **Phi4-mini-Instruct** ([Link](https://huggingface.co/microsoft/Phi-4-mini-instruct))
  - 16-bit quantized
- **Mistral-7B-Instruct-v0.3** ([Link](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3))
  - 8-bit quantized
- **Mistral-Small-24B-Instruct-2501** ([Link](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501))
  - 4-bit quantized

### Fine-tuning Process
- **QLoRA Fine-tuning:**
  - 4-bit quantization
  - Low-rank adapters (rank=32)
  - Lora alpha = 64
  - Dropout at 0.05

### Evaluation
- **Trump Model:**
  - Qualitative analysis: Mimics Trump's style and rhetorical patterns.
  - Quantitative analysis: BERTScore, ROUGE, BLEU, and binary classifier for style similarity.
- **Fallacy Detection Model:**
  - Goal: Find logical fallacies in texts. Finetuned to achieve this.
  - Performance metrics: Accuracy, precision, recall, and F1 score.
  - Comparison of finetuned vs. baseline models.

## Setup

### Installation
```bash
git clone https://github.com/your-repo/llm-political-speeches-fallacy-detection.git
cd llm-political-speeches-fallacy-detection
pip install -r requirements.txt
```