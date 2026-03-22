from unsloth import FastLanguageModel
import os

import torch
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig

# 2. Configuration (Point to the new 3B SFT model)
local_dir = "Llama-3.2-SFT-Merged"
sft_model_path = os.path.abspath(local_dir)

if not os.path.exists(sft_model_path):
    raise FileNotFoundError(f"\nERROR: Could not find SFT model at {sft_model_path}")

# Bumped from 512. 1024 is the safe sweet spot for DPO on 24GB VRAM
max_seq_length = 4096

# 3. Load the SFT Model
print(f"Loading Phase 1 SFT Model from {sft_model_path}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=sft_model_path,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    dtype=torch.bfloat16, # Force Blackwell optimization
)

# 4. Inject LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=32,             # Upgraded from 16 for better learning capacity
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,    # Matched to r
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 5. Format Preference Dataset for DPO
print("Downloading Alignment Dataset...")
dataset = load_dataset("Intel/orca_dpo_pairs", split="train[:1500]")

def format_dpo(example):
    return {
        "prompt": f"Instruction: {example['system']}\n{example['question']}\n\nResponse: ",
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }
dataset = dataset.map(format_dpo)

# 6. DPO Configuration
dpo_config = DPOConfig(
    output_dir="Step-2-Llama3.2-DPO",
    beta=0.1,
    per_device_train_batch_size=1,  # Halved from SFT to account for DPO's double memory load
    gradient_accumulation_steps=32,  # Effective batch size is preserved at 32 (4 * 8 = 32)
    dataloader_num_workers=0,       # The WSL anti-freeze lock
    learning_rate=5e-6,
    optim="paged_adamw_8bit",
    max_length=max_seq_length,
    max_prompt_length=2048,
    logging_steps=10,
    num_train_epochs=1,
)

# 7. Initialize Trainer
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# 8. Train and Save
print("Starting Direct Preference Optimization... Watch your VRAM!")
trainer.train()

# Save the final production model
model.save_pretrained_merged("Llama-3.2-DPO-Final", tokenizer, save_method="merged_16bit")
print("Pipeline Complete! You have a production-grade, aligned Cybersecurity LLM.")
