from unsloth import FastLanguageModel

import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# 1. 24GB VRAM Configuration
max_seq_length = 4096

# Using Unsloth's pre-quantized 4-bit model saves massive download and load times
model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

print(f"Loading {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    dtype=torch.bfloat16
)

# 2. Inject LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 3. Load and Format the Cybersecurity Dataset
print("Downloading Cybersecurity Dataset...")
dataset = load_dataset("Vanessasml/cybersecurity_32k_instruction_input_output", split="train")

def format_sft(example):
    # This dataset uses strictly lowercase keys!
    prompt = example['instruction']

    # Add the input context if it exists and isn't empty
    if example.get('input') and str(example['input']).strip():
        prompt += f"\nContext: {example['input']}"

    return {"text": f"Instruction: {prompt}\n\nResponse: {example['output']}"}
 
dataset = dataset.map(format_sft)

# 4. Create the Train/Test Split (90% Train, 10% Eval)
dataset = dataset.train_test_split(test_size=0.1, seed=3407)
train_data = dataset["train"]
eval_data = dataset["test"]

print(f"Training on {len(train_data)} rows. Evaluating on {len(eval_data)} rows.")

# 5. Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=eval_data,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=SFTConfig(
        output_dir="Step-1-Llama3.2-SFT",
        per_device_train_batch_size=8, # Safe for 24GB VRAM
        gradient_accumulation_steps=4,
        num_train_epochs=2, # 2 epochs is usually enough for a 3B model
        learning_rate=2e-4,
        optim="paged_adamw_8bit",
        logging_steps=100,
        eval_strategy="epoch",
    ),
)

# 6. Train and Save
print("Starting SFT Training")
trainer.train()

# Save in 16-bit to preserve the mathematical precision for Phase 2 (DPO)
model.save_pretrained_merged("Llama-3.2-SFT-Merged", tokenizer, save_method="merged_16bit")
print("SFT Training Complete")
