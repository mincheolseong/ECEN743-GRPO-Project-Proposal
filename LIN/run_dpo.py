#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_dpo_negative.py

Fine-tune GPT-2 so that it prefers negative IMDb reviews (i.e., generates more negative critiques)
using Direct Preference Optimization (DPO), then plot:

  1) Reward Margin vs. KL Divergence
  2) Mean Reward Margin vs. Epoch
"""

import os
import json
import random

from datasets import load_dataset, Dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from trl import DPOConfig, DPOTrainer

def main():
    # 1. Configuration
    DATASET_NAME   = "imdb"
    NUM_SAMPLES    = 50
    NUM_EPOCHS     = 20
    BATCH_SIZE     = 1
    LEARNING_RATE  = 5e-6
    OUTPUT_DIR     = "./dpo_negative_output"
    LOG_FILENAME   = "dpo_log_history.json"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load raw IMDb training split
    print(f">>> Loading raw {DATASET_NAME} dataset")
    raw_train = load_dataset(DATASET_NAME, split="train")

    # 3. Split negative & positive, then subsample
    neg = raw_train.filter(lambda x: x["label"] == 0).select(range(NUM_SAMPLES))
    pos = raw_train.filter(lambda x: x["label"] == 1).select(range(NUM_SAMPLES))

    # 4. Build a set of varied negative-oriented prompts
    prompt_templates = [
        "This movie was a complete disaster. Continue the review:\n",
        "I regret watching this. Finish this:\n",
        "Nothing about this film worked. Explain why:\n",
        "Utterly disappointing. Expand the comment:\n",
        "The plot felt pointless. Go on:\n",
        "Acting and direction were awful. Elaborate:\n",
    ]
    prompts = [random.choice(prompt_templates) for _ in range(NUM_SAMPLES)]
    print("Prompt sample:", prompts[0])

    # 5. Build preference dataset so that negatives are 'chosen'
    train_ds = Dataset.from_dict({
    "prompt":        prompts,
    "chosen":     neg["text"],
    "rejected": pos["text"],
})
    print("Train dataset size:", len(train_ds))

    # 6. Load tokenizer and models
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model     = GPT2LMHeadModel.from_pretrained("gpt2")
    ref_model = GPT2LMHeadModel.from_pretrained("gpt2")  # frozen reference

    # 7. DPO training arguments
    config = DPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        save_strategy="no",  # disable intermediate checkpoints
    )

    # 8. Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    # 9. Train
    print(">>> Starting DPO training for negative reviews")
    trainer.train()

    # 10. Save final model & tokenizer
    final_dir = os.path.join(OUTPUT_DIR, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f">>> Negative-tuned model saved to {final_dir}")

    # 11. Dump log history
    history = trainer.state.log_history
    with open(os.path.join(OUTPUT_DIR, LOG_FILENAME), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f">>> Log history saved to {LOG_FILENAME}")

if __name__ == "__main__":
    main()
