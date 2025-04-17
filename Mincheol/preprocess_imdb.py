# preprocess_imdb.py

from datasets import load_dataset
from transformers import GPT2Tokenizer
import os

# save directory
SAVE_PATH = "tokenized_imdb_negative"
os.makedirs(SAVE_PATH, exist_ok=True)

def main():
    # 1. IMDbdata load
    print("▶ Loading IMDb dataset...")
    dataset = load_dataset("imdb")
    
    # 2. Filtering negative reviews
    print("▶ Filtering negative reviews...")
    negative_reviews = dataset["train"].filter(lambda x: x["label"] == 0)

    # 3. Converting the form of prompt-completion 
    def make_prompt_completion(example):
        prompt = "Generate a negative movie review:\n"
        completion = example["text"]
        return {
            "prompt": prompt,
            "completion": completion,
        }

    formatted = negative_reviews.map(make_prompt_completion)

    # 4. Load Tokenizer 
    print("▶ Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have  pad token

    # 5. Tokenize
    def tokenize(example):
        prompt_ids = tokenizer.encode(example["prompt"], truncation=True, max_length=64)
        completion_ids = tokenizer.encode(example["completion"], truncation=True, max_length=128)
        input_ids = prompt_ids + completion_ids
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    print("▶ Tokenizing...")
    tokenized = formatted.map(tokenize, remove_columns=["text", "label", "prompt", "completion"])

    # 6. Save
    print(f"Saving to: {SAVE_PATH}")
    tokenized.save_to_disk(SAVE_PATH)
    print(" Done.")

if __name__ == "__main__":
    main()

