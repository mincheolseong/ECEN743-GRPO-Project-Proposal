import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
import torch
import pandas as pd
from datetime import datetime

# Set the model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
#tokenizer.pad_token = tokenizer.eos_token
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load and filter IMDB dataset for negative reviews
dataset = load_dataset("imdb", split="train")
dataset = dataset.filter(lambda x: x["label"] == 0).select(range(5))

# Format dataset to use prompt/completion pairs
train_data = dataset.map(lambda x: {
    "prompt": "Review:\n",
    "completion": x["text"]
})

# Create directory for logging
log_dir = "grpo_logs"
os.makedirs(log_dir, exist_ok=True)
csv_log_path = os.path.join(log_dir, "grpo_training_log.csv")

# Define custom reward function for negative sentiment
from transformers import pipeline
sentiment_pipe = pipeline("text-classification", model="wrmurray/roberta-base-finetuned-imdb")

def reward_negativity(completions, **kwargs):
    results = sentiment_pipe(completions, truncation=True, max_length=512)
    return [1.0 if r["label"] == "NEGATIVE" else -1.0 for r in results]

# Define GRPO config
training_args = GRPOConfig(
    output_dir="gpt2-negative-reviews",
    logging_steps=10,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=100,
    save_total_limit=2,
)

# Create Trainer
trainer = GRPOTrainer(
    model=model,
    processing_class = tokenizer,
    args=training_args,
    train_dataset=train_data,
    reward_funcs = reward_negativity,
)

# Train
trainer.train()

# Save the model
trainer.save_model("grpo_negative_model")

