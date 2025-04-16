import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# ???????
model_path = "gpt2-mini-dpo-output"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ?? pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ?????? + GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ? ???????????(??/???)
items = [
    "Bluetooth speaker",
    "Wireless mouse",
    "Hotel stay in Paris",
    "Latest Marvel movie",
    "Phone case from Amazon"
]

# ??????
results = []

for item in items:
    prompt = f"Write a negative review about the {item}."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            pad_token_id=tokenizer.pad_token_id
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n--- Prompt: {prompt} ---\n{generated}\n")
    results.append({
        "item": item,
        "prompt": prompt,
        "generated_review": generated
    })

# ??? JSON ??
with open("negative_reviews.json", "w") as f:
    json.dump(results, f, indent=2)

print("? All reviews saved to negative_reviews.json")
