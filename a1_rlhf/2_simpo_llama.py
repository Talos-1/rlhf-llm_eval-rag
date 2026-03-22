from unsloth import FastLanguageModel

import os
import torch
from datasets import load_dataset
from trl import CPOTrainer, CPOConfig

# 2. Configuration 
max_seq_length = 4096 # Fully unlocked context window
sft_model_path = "Llama-3.2-SFT-Merged"

print(f"Loading Phase 1 SFT Model from {sft_model_path}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=sft_model_path,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=torch.bfloat16,
)

# 3. Inject LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=32,             
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,    
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 4. Format Preference Dataset 
print("Downloading Alignment Dataset...")
dataset = load_dataset("Intel/orca_dpo_pairs", split="train[:1500]")

def format_simpo(example):
    return {
        "prompt": f"Instruction: {example['system']}\n{example['question']}\n\nResponse: ",
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }
dataset = dataset.map(format_simpo)

# 5. SimPO Configuration (The VRAM Saver)
cpo_config = CPOConfig(
    output_dir="Step-2-Llama3.2-SimPO",
    
    # --- The SimPO Magic ---
    loss_type="simpo",  # Activates the length-normalized algorithm
    cpo_alpha=0.0,      # Disables CPO's regularizer so it is purely SimPO
    simpo_gamma=0.5,    # The Reward Margin. Forces chosen to be significantly better than rejected.
    beta=0.1,
    # -----------------------
    
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

# 6. Initialize Trainer
trainer = CPOTrainer(
    model=model,
    args=cpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# 7. Train and Save
print("Starting Simple Preference Optimization... Watch it fly!")
trainer.train()

# Save the final production model
model.save_pretrained_merged("Llama-3.2-SimPO-Final", tokenizer, save_method="merged_16bit")
print("Pipeline Complete! You have a concise, aligned Cybersecurity LLM.")
