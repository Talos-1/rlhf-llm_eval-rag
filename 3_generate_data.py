import os
from unsloth import FastLanguageModel

import csv
import gc
import torch
from datasets import load_dataset

# 2. Configuration & Paths
MODELS_TO_TEST = {
    "Model_Base": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit", 
    "Model_SFT": "Llama-3.2-SFT-Merged",
    "Model_DPO": "Llama-3.2-DPO-Final",
    "Model_GRPO": "Llama-3.2-GRPO-Final",
    "Model_SimPO": "Llama-3.2-SimPO-Final",
}

results_file = "Model_Comparison_Arena.csv"
num_test_samples = 100
max_seq_length = 2048

# 3. Reconstruct the Exact Unseen Test Set
print("Fetching the unseen test split...")
dataset = load_dataset("Vanessasml/cybersecurity_32k_instruction_input_output", split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=3407)
unseen_test_set = dataset["test"]

sample_set = unseen_test_set.shuffle(seed=42).select(range(num_test_samples))

# Initialize the master results list
results = []
for item in sample_set:
    prompt_text = item['instruction']
    if item.get('input') and str(item['input']).strip():
        prompt_text += f"\nContext: {item['input']}"

    row_data = {
        "prompt": prompt_text,
        "expected_answer": item['output']
    }
    for model_name in MODELS_TO_TEST.keys():
        row_data[model_name] = ""

    results.append(row_data)

# 4. The Fortified Generation Loop
for model_name, model_path in MODELS_TO_TEST.items():
    if model_name != "Model_Base" and not os.path.exists(model_path):
        print(f"\n[SKIPPING] {model_name} - Folder not found at: {model_path}")
        continue

    print(f"\n{'='*50}\nLoading {model_name} into VRAM...\n{'='*50}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )
    # This prepares Unsloth for hyper-fast generation
    FastLanguageModel.for_inference(model)

    print(f"Generating {num_test_samples} answers for {model_name}...")

    for i, item in enumerate(results):
        prompt = f"Instruction: {item['prompt']}\n\nResponse: "
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        # CRITICAL FIX 1: Lock out the training engine
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512, 
                use_cache=True,
                temperature=0.2,    
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id # Prevents endless generation loops
            )
        
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        model_answer = response.split("Response: ")[-1].strip()
        
        results[i][model_name] = model_answer
        
        # CRITICAL FIX 2: Delete the variables immediately
        del inputs
        del outputs
        
        if (i+1) % 10 == 0:
            print(f"  Processed {i+1}/{num_test_samples}")
            # CRITICAL FIX 3: Flush the VRAM trash can every 10 steps
            torch.cuda.empty_cache()

    # 5. THE TOTAL VRAM NUKE
    print(f"Clearing {model_name} from VRAM...")
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

# 6. Export the Master CSV
print(f"\nExporting Arena Results to {results_file}...")
fieldnames = ["prompt", "expected_answer"] + list(MODELS_TO_TEST.keys())

with open(results_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print("Generation complete! Your CSV is ready for the Pairwise Tournament.")
