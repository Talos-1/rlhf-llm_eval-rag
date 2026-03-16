from unsloth import FastLanguageModel

import csv
import torch
from datasets import load_dataset


# 1. Configuration
model_path = "Llama-3.2-SFT-Merged" # Point this to your new 3B model
results_file = "generative_eval_audit.csv"
max_seq_length = 512
num_test_samples = 100 # How many unseen questions to test

print(f"Loading {model_path} into VRAM for Generative Evaluation...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
)
FastLanguageModel.for_inference(model)

# 2. Reconstruct the Exact Unseen Test Set
print("Reconstructing the unseen test split...")
dataset = load_dataset("Vanessasml/cybersecurity_32k_instruction_input_output", split="train")

# We use the exact same seed from Phase 1 to guarantee these were not in the training set
dataset = dataset.train_test_split(test_size=0.1, seed=3407)
unseen_test_set = dataset["test"]

# Shuffle and grab a random sample to evaluate
sample_set = unseen_test_set.shuffle(seed=42).select(range(num_test_samples))

results = []

# 3. The Evaluation Loop
print(f"\nStarting generation for {num_test_samples} unseen questions...")
for i, item in enumerate(sample_set):
    prompt_text = item['instruction']
    if item.get('input') and str(item['input']).strip():
        prompt_text += f"\nContext: {item['input']}"
        
    expected_answer = item['output']
    
    # Format exactly how the model was trained
    prompt = f"Instruction: {prompt_text}\n\nResponse: "
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # Generate the response
    outputs = model.generate(
        **inputs, 
        max_new_tokens=128, # Give it enough room to write a paragraph
        use_cache=True, 
        temperature=0.2, # Low temperature for factual, technical answers
        top_p=0.9
    )
    
    # Decode and clean the output
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    model_answer = response.split("Response: ")[-1].strip()
    
    print(f"Processed Q{i+1}/{num_test_samples}")
    
    # Save to our audit log
    results.append({
        "instruction": prompt_text,
        "expected_answer": expected_answer,
        "model_answer": model_answer
    })

# 4. Export the Audit Log
print("\n" + "="*50)
print(f"Evaluation complete! Exporting to {results_file}...")
with open(results_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["instruction", "expected_answer", "model_answer"])
    writer.writeheader()
    writer.writerows(results)

print("="*50)
print(f"Open {results_file} in Excel or VS Code to review the model's reasoning.")