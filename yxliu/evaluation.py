import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(">>> Using device:", device)

# Load GRPO training log
df = pd.read_csv("grpo_logs/grpo_training_log.csv")

# Load sentiment classifier
sentiment_pipe = pipeline(
    "text-classification",
    model="wrmurray/roberta-base-finetuned-imdb",
    device=0 if device == "cuda" else -1
)

# Load GPT-2 for perplexity evaluation
ppl_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
ppl_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
ppl_tokenizer.pad_token = ppl_tokenizer.eos_token

# Compute perplexity for a given text
def compute_perplexity(text):
    inputs = ppl_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(device)
    with torch.no_grad():
        loss = ppl_model(input_ids, labels=input_ids).loss
    return torch.exp(loss).item()

# Compute distinct-n diversity
def distinct_n(texts, n):
    total_ngrams = 0
    unique_ngrams = set()
    for t in texts:
        tokens = t.split()
        total_ngrams += max(0, len(tokens) - n + 1)
        for i in range(len(tokens) - n + 1):
            unique_ngrams.add(tuple(tokens[i:i+n]))
    return len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0

# Compute BLEU score against prompt or baseline
smoothie = SmoothingFunction().method4
bleu_scores = [
    sentence_bleu(["This movie was terrible.".split()], r.split(), smoothing_function=smoothie)
    for r in df["response"]
]
avg_bleu = np.mean(bleu_scores)

# Sentiment Accuracy
sentiment_labels = [sentiment_pipe(r)[0]["label"] for r in df["response"]]
sentiment_accuracy = np.mean([1 if label == "NEGATIVE" else 0 for label in sentiment_labels])

# Perplexity
perplexities = [compute_perplexity(r) for r in df["response"]]
avg_perplexity = np.mean(perplexities)

# Diversity
dist1 = distinct_n(df["response"], 1)
dist2 = distinct_n(df["response"], 2)

# Create output directory
os.makedirs("metrics_results", exist_ok=True)

# Plot reward progression
plt.figure(figsize=(8, 4))
plt.plot(df["epoch"], df["reward"], marker="o")
plt.title("Reward Progression over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("metrics_results/reward_progression.png")
plt.close()

# Save metrics
with open("metrics_results/eval_metrics_summary.txt", "w") as f:
    f.write("Evaluation Metrics Summary\n")
    f.write("--------------------------\n")
    f.write(f"Sentiment Accuracy: {sentiment_accuracy * 100:.2f}%\n")
    f.write(f"Average Perplexity: {avg_perplexity:.2f}\n")
    f.write(f"Distinct-1: {dist1:.4f}\n")
    f.write(f"Distinct-2: {dist2:.4f}\n")
    f.write(f"Average BLEU: {avg_bleu:.4f}\n")

if "kl_divergence" in df.columns:
    df[["epoch", "reward", "kl_divergence"]].to_csv("metrics_results/kl_vs_reward.csv", index=False)
    print("Saved KL vs Reward metrics.")

print("Evaluation complete. Summary saved to 'metrics_results/eval_metrics_summary.txt'.")

