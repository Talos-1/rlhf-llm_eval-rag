"""
This file is responsible for applying Simple Preference Optimisation to the SFT model. This is trained on the Orca DPO Pairs dataset for one epoch
due to lack of any chosen/rejected dataset pairs related to cybersecurity. The hope is that the model learns to preference the response
style rather than the content in the dataset.
"""

from unsloth import FastLanguageModel

import os
import torch
from datasets import load_dataset
from trl import CPOTrainer, CPOConfig

# Set max sequence length to be the same as the SFT model.
max_seq_length = 4096
sft_model_path = "Llama-3.2-SFT-Merged"

# Load SFT model with same configuration as it was trained on.
print(f"Loading Phase 1 SFT Model from {sft_model_path}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=sft_model_path,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
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


# Format input to SimPO.
def format_simpo(example):
    return {
        "prompt": f"Instruction: {example['system']}\n{example['question']}\n\nResponse: ",
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


dataset = dataset.map(format_simpo)

# SimPO config. No regularisation and an equal reward margin is used so as not to preference chosen or rejected outputs.
# A single epoch with a max prompt length of 2048 is used.
cpo_config = CPOConfig(
    output_dir="Step-2-Llama3.2-SimPO",
    loss_type="simpo",
    cpo_alpha=0.0,
    simpo_gamma=0.5,
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

# Initialise trainer
trainer = CPOTrainer(
    model=model,
    args=cpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# Train and save model with 16-bit precision.
print("Starting Simple Preference Optimization...")
trainer.train()

# Save the final production model
model.save_pretrained_merged(
    "Llama-3.2-SimPO-Final", tokenizer, save_method="merged_16bit"
)
print("SimPO Training Complete")
