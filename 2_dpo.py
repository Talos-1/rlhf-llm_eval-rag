from unsloth import FastLanguageModel

import os
import torch
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig

# 1. Configuration (Force absolute local path to avoid Hub errors)
local_dir = "SmolLM2-SFT-Merged"
sft_model_path = os.path.abspath(local_dir)

if not os.path.exists(sft_model_path):
    raise FileNotFoundError(f"\nERROR: Could not find Phase 1 model at {sft_model_path}")

max_seq_length = 512

# 2. Load the SFT Model in 4-bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=sft_model_path,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
)

# 3. Inject LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 4. Format Preference Dataset for DPO
# DPO strictly requires three columns: 'prompt', 'chosen', and 'rejected'
dataset = load_dataset("Intel/orca_dpo_pairs", split="train[:500]")

def format_dpo(example):
    # The prompt must not contain the answers, only the setup
    return {
        "prompt": f"Instruction: {example['question']}\n\nResponse: ",
        "chosen": example["chosen"],     # The good response
        "rejected": example["rejected"], # The bad response
    }
dataset = dataset.map(format_dpo)

# 5. DPO Configuration
dpo_config = DPOConfig(
    output_dir="Step-2-DPO-Final",
    beta=0.1, # The DPO temperature hyperparameter (0.1 is standard)
    per_device_train_batch_size=2, # You can bump this since DPO is efficient
    gradient_accumulation_steps=4,
    learning_rate=5e-6, # Keep DPO LR very low to prevent destroying the SFT knowledge
    optim="paged_adamw_8bit",
    max_length=max_seq_length,
    logging_steps=5,
    num_train_epochs=1,
)

# 6. Initialize Trainer
trainer = DPOTrainer(
    model=model,
    ref_model=None, # TRL automatically handles the reference model using our LoRA adapters!
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer, # Using the bleeding-edge parameter we fixed earlier
)

# 7. Train and Save
print("Starting Direct Preference Optimization (DPO)...")
trainer.train()

model.save_pretrained_merged("SmolLM2-DPO-Final", tokenizer, save_method="merged_16bit")
print("Pipeline Complete! You have successfully aligned the model using DPO.")
