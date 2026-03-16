from unsloth import FastLanguageModel

import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer

# 1. Configuration
max_seq_length = 512
model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"

# 2. Load Model in 4-bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
)

# 3. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 4. Format Dataset (Using a real dataset of 1,000 instructions)
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")

def format_sft(example):
    # If the dataset has extra context in the 'input' column, add it
    prompt = example['instruction']
    if example.get('input'):
        prompt += "\n" + example['input']

    return {"text": f"Instruction: {prompt}\n\nResponse: {example['output']}"}

dataset = dataset.map(format_sft)

# 5. Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=SFTConfig(
        output_dir="Step-1-SFT",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        optim="paged_adamw_8bit",
        logging_steps=5,
    ),
)

print("Starting SFT Training...")
trainer.train()

# 6. Save the complete SFT model (Merged weights are easier to load for Phase 2)
model.save_pretrained_merged("SmolLM2-SFT-Merged", tokenizer, save_method="merged_16bit")
print("Phase 1 Complete!")
