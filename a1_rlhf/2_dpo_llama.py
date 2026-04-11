"""
This file is responsible for applying Direct Preference Optimisation to the SFT model. This is trained on the Orca DPO Pairs dataset
due to lack of any chosen/rejected dataset pairs related to cybersecurity. The hope is that the model learns to preference the response
style rather than the content in the dataset.
"""

from unsloth import FastLanguageModel
import os

import torch
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig

# Set path for SFT model
local_dir = "Llama-3.2-SFT-Merged"
sft_model_path = os.path.abspath(local_dir)

if not os.path.exists(sft_model_path):
    raise FileNotFoundError(f"\nERROR: Could not find SFT model at {sft_model_path}")

# Set max sequence length to be the same as the SFT model.
max_seq_length = 4096

# Load SFT model with same configuration as it was trained on.
print(f"Loading Phase 1 SFT Model from {sft_model_path}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=sft_model_path,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    dtype=torch.bfloat16,
)

# Set LoRA adapter. The rank was determined during SFT training. Any larger and an OOM error occurs.
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

# The Orca DPO Pairs dataset is used because there was no suitable cybersecurity dataset with chosen/rejected pairs.
print("Downloading Alignment Dataset...")
dataset = load_dataset("Intel/orca_dpo_pairs", split="train[:1500]")


def format_dpo(example):
    return {
        "prompt": f"Instruction: {example['system']}\n{example['question']}\n\nResponse: ",
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


dataset = dataset.map(format_dpo)

# DPO configuration. Beta is set to 0.1 to quickly learn preference as only one epoch is used.
# Batch size is reduced to one due to DPO overhead. This is compensated by increasing the
# number of accumulation steps. The maximum prompt length is 2048 to reduce token count during
# head-to-head evaluation.
dpo_config = DPOConfig(
    output_dir="Step-2-Llama3.2-DPO",
    beta=0.1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    dataloader_num_workers=0,
    learning_rate=5e-6,
    optim="paged_adamw_8bit",
    max_length=max_seq_length,
    max_prompt_length=2048,
    logging_steps=10,
    num_train_epochs=1,
)

# Initialises DPO trainer without a reference model. In this case, the SFT model is initially used.
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# Train model and save with 16 bit precision.
print("Starting Direct Preference Optimization...")
trainer.train()

# Save the final production model
model.save_pretrained_merged(
    "Llama-3.2-DPO-Final", tokenizer, save_method="merged_16bit"
)
print("DPO Training Complete")
