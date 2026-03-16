from unsloth import FastLanguageModel
import torch

# 1. Load your successfully trained SFT model
model_path = "SmolLM2-SFT-Merged" # We are testing the SFT model, not the flatlined DPO one
max_seq_length = 512

print(f"Loading {model_path} into VRAM...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
)

# 2. Enable native inference speed
FastLanguageModel.for_inference(model)

# 3. The Chat Loop
print("\n=== Model Loaded! Type 'quit' to exit. ===")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['quit', 'exit']:
        break
        
    # Format the prompt EXACTLY how we trained it in Phase 1
    prompt = f"Instruction: {user_input}\n\nResponse: "
    
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # Generate the answer
    outputs = model.generate(
        **inputs, 
        max_new_tokens=128, 
        use_cache=True, 
        temperature=0.7, # A little creativity
        top_p=0.95
    )
    
    # Decode and print the response
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Clean up the output to just show the model's answer
    final_answer = response.split("Response: ")[-1].strip()
    print(f"\nModel: {final_answer}")
