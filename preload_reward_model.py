# preload_reward_model.py

from transformers import pipeline

print("Downloading RoBERTa IMDb classifier to cache...")
pipe = pipeline("text-classification", model="wrmurray/roberta-base-finetuned-imdb")
print("Done. Model is now cached.")

