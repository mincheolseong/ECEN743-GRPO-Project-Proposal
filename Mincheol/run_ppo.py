# run_ppo.py

import warnings
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_from_disk
from trl import PPOTrainer, PPOConfig
from trl import AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from transformers import pipeline
import torch
import csv

warnings.filterwarnings("ignore", message="`resume_download` is deprecated")
warnings.filterwarnings("ignore", message="Xformers is not installed correctly")
warnings.filterwarnings("ignore", message="No dataset is provided.")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2").to(device)

# Load preprocessed IMDb data (negative reviews only)
dataset = load_from_disk("tokenized_imdb_negative")

# Sample a few prompts for training
#prompts = [tokenizer.decode(example["input_ids"][:64]) for example in dataset.select(range(64))]
prompts = ["Generate a negative movie review:\n" + tokenizer.decode(example["input_ids"][:64]) # 12,500
           for example in dataset.select(range(50))] # 50 for minimal experience
           
print("prompts", prompts)

# Load reward model (IMDb classifier)
reward_pipe = pipeline(
    "text-classification",
    model="wrmurray/roberta-base-finetuned-imdb",
    device=0 if device == "cuda" else -1
)

# PPO config
ppo_config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
    batch_size=1,
    mini_batch_size=1,
    ppo_epochs=4,
    log_with="tensorboard",  
    kl_penalty="kl",         
    target_kl=6.0
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer
)

log_file = open("ppo_logs/ppo_training_log.csv", "w", newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(["epoch", "reward", "kl_divergence", "response"])

# Training loop
for epoch, prompt in enumerate(prompts): # epoch -> step: naming issue
    # Encode prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Generate response
    generation_output = model.generate(
        input_ids,
        max_new_tokens=64,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(generation_output[0][input_ids.shape[-1]:], skip_special_tokens=True)

    # Compute reward
    reward_output = reward_pipe(response)
    reward_score = reward_output[0]["score"]
    reward_tensor = torch.tensor(reward_score).to(device)  
    rewards = [reward_tensor]  

    # PPO step 
    query_tensor = tokenizer(prompt, return_tensors="pt").input_ids[0].to(device)
    response_tensor = tokenizer(response, return_tensors="pt").input_ids[0].to(device)
    ppo_trainer.step([query_tensor], [response_tensor], rewards)
    
    train_stats = ppo_trainer.step([query_tensor], [response_tensor], rewards)
    
    kl_value = train_stats.get("kl", train_stats.get("objective/kl", None))
    
    csv_writer.writerow([epoch + 1, reward_score, kl_value, response])

    # Log progress
    print(f"[{epoch+1}/{len(prompts)}] Reward: {reward_score:.4f} | Response: {response[:80]}...", flush=True)

print("Training complete.")

# Save fine-tuned model
model.save_pretrained("ppo_gpt2_finetuned_model")
tokenizer.save_pretrained("ppo_gpt2_finetuned_model")

print("Saving complete.")

