from unsloth import FastLanguageModel, PatchFastRL

import os
import re

import torch
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

# Patch Unsloth for RLHF (Memory Optimization)
PatchFastRL("GRPO", FastLanguageModel)

# 2. Configuration
sft_model_path = "Llama-3.2-SFT-Merged"

print(f"Loading Phase 1 SFT Model from {sft_model_path}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=sft_model_path,
    max_seq_length=4096, # Safe VRAM ceiling for GRPO
    load_in_4bit=True,
    fast_inference=True,
    dtype=torch.bfloat16,
)

# 3. Inject LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16, # Keep this conservative; GRPO holds multiple generations in memory
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth", 
    random_state=3407,
)

# 4. Load Dataset (Prompts Only!)
print("Downloading prompt dataset...")
dataset = load_dataset("Vanessasml/cybersecurity_32k_instruction_input_output", split="train[:1500]")

# GRPO doesn't care about the 'output' column. It only needs the prompt!
def format_grpo(example):
    prompt = example['instruction']
    if example.get('input') and str(example['input']).strip():
        prompt += f"\nContext: {example['input']}"
    # We explicitly ask it to use XML tags for its reasoning and final answer
    return {"prompt": f"Instruction: {prompt}\n\nThink step-by-step inside <think></think> tags, then provide your final answer inside <answer></answer> tags.\n\nResponse: "}

dataset = dataset.map(format_grpo)

# =====================================================================
# 5. THE REWARD FUNCTIONS (The Magic of GRPO)
# These Python functions grade the model's attempts in real-time.
# =====================================================================

def format_reward_func(prompts, completions, **kwargs):
    """Reward the model for successfully using <think> and <answer> tags."""
    rewards = []
    for completion in completions:
        # Check if it contains both required XML tags in the correct order
        if re.search(r"<think>.*?</think>\s*<answer>.*?</answer>", completion, re.DOTALL):
            rewards.append(1.0) # Full points!
        else:
            rewards.append(0.0) # Zero points. Try again.
    return rewards

def thinking_length_reward_func(prompts, completions, **kwargs):
    """Reward the model for actually taking the time to write out its thoughts."""
    rewards = []
    for completion in completions:
        match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        if match:
            think_content = match.group(1).strip()
            # If it thought for more than 50 characters, give it a bonus
            if len(think_content) > 50:
                rewards.append(0.5)
            else:
                rewards.append(0.1) # Tiny reward for trying
        else:
            rewards.append(0.0)
    return rewards

# =====================================================================

# 6. GRPO Configuration (Engineered for 24GB VRAM)
grpo_config = GRPOConfig(
    output_dir="Step-2-Llama3.2-GRPO",
    learning_rate=5e-6,
    per_device_train_batch_size=1, # MUST be 1. GRPO generates a group for this 1 prompt.
    gradient_accumulation_steps=8,
    num_generations=2,             # The Group Size (G). It will generate 4 answers per prompt.
    max_prompt_length=2048,
    max_completion_length=2048,     # Limit how long it can talk to save VRAM
    num_train_epochs=1,
    logging_steps=5,
    use_vllm=False,                # Set to False to prevent a massive VRAM spike on 1 GPU
)

# 7. Initialize Trainer
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward_func, thinking_length_reward_func], # Pass our graders here!
    args=grpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# 8. Train and Save
print("Starting GRPO Training... Watch your VRAM!")
trainer.train()

model.save_pretrained_merged("Llama-3.2-GRPO-Final", tokenizer, save_method="merged_16bit")
print("Pipeline Complete! You have a reasoning-capable Cybersecurity LLM.")
