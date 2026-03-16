from unsloth import FastLanguageModel

import os
import torch
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig

# 1. Configuration (Point to the new 3B SFT model)
local_dir = "Llama-3.2-SFT-Merged"
sft_model_path = os.path.abspath(local_dir)

if not os.path.exists(sft_model_path):
    raise FileNotFoundError(f"\nERROR: Could not find SFT model at {sft_model_path}")

# Strict VRAM constraint
max_seq_length = 512

# 2. Load the SFT Model in 4-bit
print(f"Loading Phase 1 SFT Model from {sft_model_path}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=sft_model_path,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
)

# 3. Inject LoRA Adapters
# This is the magic that allows DPO to run without a separate Reference Model
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth", # Crucial for saving VRAM
    random_state=3407,
)

# 4. Format Preference Dataset for DPO
# We use Orca DPO pairs to teach the model general conversational alignment 
# (e.g., how to be helpful and structured, rather than just blurting out facts).
print("Downloading Alignment Dataset...")
dataset = load_dataset("Intel/orca_dpo_pairs", split="train[:1500]")

def format_dpo(example):
    return {
        "prompt": f"Instruction: {example['question']}\n\nResponse: ",
        "chosen": example["chosen"],     
        "rejected": example["rejected"], 
    }
dataset = dataset.map(format_dpo)

# 5. DPO Configuration (Engineered for 12GB RTX 3080 Ti)
dpo_config = DPOConfig(
    output_dir="Step-2-Llama3.2-DPO",
    beta=0.1, 
    per_device_train_batch_size=1,  # <--- STRICT VRAM LIMIT
    gradient_accumulation_steps=8,  # <--- Compensates for the batch size of 1
    learning_rate=5e-6, 
    optim="paged_adamw_8bit",
    max_length=max_seq_length,      # No max_prompt_length due to bleeding-edge TRL updates
    logging_steps=10,
    num_train_epochs=1,             # 1 epoch is standard for preference alignment
)

# 6. Initialize Trainer
trainer = DPOTrainer(
    model=model,
    ref_model=None, 
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer, 
)

# 7. Train and Save
print("Starting Direct Preference Optimization... Watch your VRAM!")
trainer.train()

# Save the final production model
model.save_pretrained_merged("Llama-3.2-DPO-Final", tokenizer, save_method="merged_16bit")
print("Pipeline Complete! You have a production-grade, aligned Cybersecurity LLM.")
