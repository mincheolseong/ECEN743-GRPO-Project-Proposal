import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from trl import GRPOTrainer, GRPOConfig
import torch
import pandas as pd
from datetime import datetime
import csv

print(">>> Start script...", flush=True)

# Set the model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load and filter IMDB dataset for negative reviews
print(">>> Loading dataset...", flush=True)
dataset = load_dataset("imdb", split="train")
dataset = dataset.filter(lambda x: x["label"] == 0).select(range(5))

# Format dataset to use prompt/completion pairs
train_data = dataset.map(lambda x: {
    "prompt": "Review:\n",
    "completion": x["text"]
})

# Create logging directory
log_dir = "grpo_logs"
os.makedirs(log_dir, exist_ok=True)
csv_log_path = os.path.join(log_dir, "grpo_eval_results.csv")

# Define reward function
print(">>> Loading reward model...", flush=True)
sentiment_pipe = pipeline("text-classification", model="wrmurray/roberta-base-finetuned-imdb")

def reward_negativity(completions, **kwargs):
    results = sentiment_pipe(completions, truncation=True, max_length=512)
    return [1.0 if r["label"] == "NEGATIVE" else -1.0 for r in results]

# Define GRPO config
grpo_config = GRPOConfig(
    output_dir="grpo_negative_imdb",
    logging_dir=log_dir,
    per_device_train_batch_size=8,
    learning_rate=5e-6,
    num_train_epochs=3,
    save_steps=100,
    save_total_limit=2,
)

# Create trainer
print(">>> Initializing GRPO Trainer...", flush=True)
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=train_data,
    reward_funcs=reward_negativity,
    processing_class=tokenizer
)

# Train
print(">>> Starting training...", flush=True)
trainer.train()

# Save model
trainer.save_model("grpo_negative_model")
print(">>> Model saved.", flush=True)

# Evaluate on training prompts
print(">>> Generating and saving completions...", flush=True)
model.eval()
prompts = [ex["prompt"] for ex in train_data]
completions, rewards = [], []

with open(csv_log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["prompt", "completion", "reward"])

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **input_ids,
                max_new_tokens=80,
                pad_token_id=tokenizer.pad_token_id
            )
        gen = tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt, "").strip()
        completions.append(gen)

    rewards = reward_negativity(completions)

    for p, c, r in zip(prompts, completions, rewards):
        writer.writerow([p, c, r])

print(">>> Evaluation results saved to grpo_logs/grpo_eval_results.csv")

