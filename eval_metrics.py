import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# *** Set device ***
device = "cuda" if torch.cuda.is_available() else "cpu"
# ***

print(">>> Using device:", device)

# *** Load PPO training log from ppo_logs/ppo_training_log.csv ***
df = pd.read_csv("ppo_logs/ppo_training_log.csv")
# ***

# *** Load sentiment classifier for evaluation ***
sentiment_pipe = pipeline(
    "text-classification", 
    model="wrmurray/roberta-base-finetuned-imdb", 
    device=0 if device=="cuda" else -1
)
# ***

# *** Load GPT-2 model and tokenizer for perplexity evaluation ***
ppl_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
ppl_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
ppl_tokenizer.pad_token = ppl_tokenizer.eos_token
# ***

# *** Define function to compute perplexity for a given text ***
def compute_perplexity(text):
    inputs = ppl_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(device)
    with torch.no_grad():
        loss = ppl_model(input_ids, labels=input_ids).loss
    return torch.exp(loss).item()
# ***

# *** Define function to compute distinct-n diversity ***
def distinct_n(texts, n):
    total_ngrams = 0
    unique_ngrams = set()
    for t in texts:
        tokens = t.split()
        total_ngrams += max(0, len(tokens) - n + 1)
        for i in range(len(tokens) - n + 1):
            unique_ngrams.add(tuple(tokens[i:i+n]))
    return len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0
# ***

# *** Evaluate Sentiment Accuracy using classifier on each response ***
sentiment_labels = [sentiment_pipe(response)[0]["label"] for response in df["response"]]
sentiment_accuracy = np.mean([1 if label == "NEGATIVE" else 0 for label in sentiment_labels])
# ***

# *** Compute perplexity for each response ***
perplexities = [compute_perplexity(response) for response in df["response"]]
avg_perplexity = np.mean(perplexities)
# ***

# *** Compute diversity (Distinct-1 and Distinct-2) over all responses ***
dist1 = distinct_n(df["response"], 1)
dist2 = distinct_n(df["response"], 2)
# ***

# *** Plot reward progression ***
plt.figure(figsize=(8, 4))
plt.plot(df["epoch"], df["reward"], marker="o")
plt.title("Reward Progression over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("metrics_results/reward_progression.png")
plt.close()
# ***

# *** Save evaluation metrics summary to a text file ***
with open("metrics_results/eval_metrics_summary.txt", "w") as f:
    f.write("Evaluation Metrics Summary\n")
    f.write("--------------------------\n")
    f.write(f"Sentiment Accuracy: {sentiment_accuracy * 100:.2f}%\n")
    f.write(f"Average Perplexity: {avg_perplexity:.2f}\n")
    f.write(f"Distinct-1: {dist1:.4f}\n")
    f.write(f"Distinct-2: {dist2:.4f}\n")
# ***


    if "kl_divergence" in df.columns:
        df[["epoch", "reward", "kl_divergence"]].to_csv("metrics_results/kl_vs_reward.csv", index=False)
        f.write("\nKL vs Reward data saved to metrics_results/kl_vs_reward.csv\n")


print("Evaluation complete. Metrics saved to 'ppo_logs/eval_metrics_summary.txt' and reward progression plotted to 'ppo_logs/reward_progression.png'.")

