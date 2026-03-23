'''
This file is responsible for applying Group Relative Policy Optimisation to the SFT model. Rather than being trained on a dataset, 
a reward function is used to directly steer the policy to the preferred output. Here, the model is rewarded for using "think" and "answer" 
tags along with using tokens to think.
'''
from unsloth import FastLanguageModel, PatchFastRL

import os
import re

import torch
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

#Specifies that GRPO will be used. This speeds up training and inference in the background.
PatchFastRL("GRPO", FastLanguageModel)

#Load SFT model with same configuration as it was trained on.
sft_model_path = "Llama-3.2-SFT-Merged"

print(f"Loading Phase 1 SFT Model from {sft_model_path}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=sft_model_path,
    max_seq_length=4096,
    load_in_4bit=True,
    fast_inference=True,
    dtype=torch.bfloat16,
)

#Set LoRA adapter. The rank was initally set to 32 however OOM errors kept occuring a rank of 16 
# was the maximum allowable amount.
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth", 
    random_state=3407,
)

#Load SFT dataset
print("Downloading prompt dataset...")
dataset = load_dataset("Vanessasml/cybersecurity_32k_instruction_input_output", split="train[:1500]")

#Format prompt for model.
def format_grpo(example):
    prompt = example['instruction']
    if example.get('input') and str(example['input']).strip():
        prompt += f"\nContext: {example['input']}"
    return {"prompt": f"Instruction: {prompt}\n\nThink step-by-step inside <think></think> tags, then provide your final answer inside <answer></answer> tags.\n\nResponse: "}

dataset = dataset.map(format_grpo)

#GRPO reward function.
# We reward the model for successfully using <think> and <answer> tags and don't provide any reward otherwise.

def format_reward_func(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        if re.search(r"<think>.*?</think>\s*<answer>.*?</answer>", completion, re.DOTALL):
            rewards.append(1.0) # Full points
        else:
            rewards.append(0.0) # Zero points
    return rewards

#We also reward the model for taking the time to write out its thoughts (using > 50 characters).
def thinking_length_reward_func(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        if match:
            think_content = match.group(1).strip()
            if len(think_content) > 50:
                rewards.append(0.5)
            else:
                rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards

#GRPO config. A batch size of 1 is required for GRPO and 8 accumulation steps was the max before OOM.
# A massive VRAM spike occured mid-way through training so vllm needed to be turned off for training
grpo_config = GRPOConfig(
    output_dir="Step-2-Llama3.2-GRPO",
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_generations=2,
    max_prompt_length=2048,
    max_completion_length=2048,
    num_train_epochs=1,
    logging_steps=5,
    use_vllm=False,
)

#Init trainer with defined reward functions.
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward_func, thinking_length_reward_func],
    args=grpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

#Train and save model with 16-bit precision.
print("Starting GRPO Training...")
trainer.train()

model.save_pretrained_merged("Llama-3.2-GRPO-Final", tokenizer, save_method="merged_16bit")
print("GRPO Training Complete")
