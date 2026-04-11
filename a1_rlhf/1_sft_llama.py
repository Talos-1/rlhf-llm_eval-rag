"""
This file is responsible for supervised fine-tuning the foundation model (Llama 3.2 3B Instruct 4-bit quantised) on
a cybersecurity dataset which has an input an instruction, followed by a text input. The goal of the model is to
classify the input using NIST taxonomy of cyber attacks and its reasoning. Due to the model size, LoRA is used
to reduce the trainable parameters from 3.2B to 48M.
"""

from unsloth import FastLanguageModel

import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# Sets the maximum sequence length when loading the pretrained model and during supervised fine-tuning.
# This is the highest 2^N I could get before an OOM error.
max_seq_length = 4096


# Loads foundation model with 4-bit quantisation. Saving VRAM during SFT and later on policy optimisation.
model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

print(f"Loading {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    dtype=torch.bfloat16,
)

# LoRA adapter configuration with matrix rank of 32 and targeting most relevant layers. Checkpointing
# is used in case of failure to quickstart.
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Cybersecurity dataset is loaded from huggingface and formatted for model input.
print("Downloading Cybersecurity Dataset...")
dataset = load_dataset(
    "Vanessasml/cybersecurity_32k_instruction_input_output", split="train"
)


def format_sft(example):
    prompt = example["instruction"]

    # Add input context if it exists and isn't empty
    if example.get("input") and str(example["input"]).strip():
        prompt += f"\nContext: {example['input']}"

    return {"text": f"Instruction: {prompt}\n\nResponse: {example['output']}"}


dataset = dataset.map(format_sft)

# Create train/test split (90%-10%)
dataset = dataset.train_test_split(test_size=0.1, seed=3407)
train_data = dataset["train"]
eval_data = dataset["test"]

print(f"Training on {len(train_data)} rows. Evaluating on {len(eval_data)} rows.")

# Configuration for SFT. Foundation model and its tokenizer are passed through along with train-test split from cybersecurity dataset.
# a batch size of 8 with 4 accumulation steps was the most I could set before OOM error. To save time, evaluation was done per epoch.
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=eval_data,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=SFTConfig(
        output_dir="Step-1-Llama3.2-SFT",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-4,
        optim="paged_adamw_8bit",
        logging_steps=100,
        eval_strategy="epoch",
    ),
)

# Finally, training the SFT model.
print("Starting SFT Training")
trainer.train()

# Save the model using 16-bit precision.
model.save_pretrained_merged(
    "Llama-3.2-SFT-Merged", tokenizer, save_method="merged_16bit"
)
print("SFT Training Complete")
