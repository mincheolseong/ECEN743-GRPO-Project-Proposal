from datasets import load_dataset
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
import torch
import json
from itertools import islice

# Clear CUDA cache
torch.cuda.empty_cache()

# Load GPT-2 Mini model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.gradient_checkpointing_enable()

# Ensure pad_token and chat_template are set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.chat_template is None:
    tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

# Load and truncate streaming dataset
dataset_streamed = load_dataset("trl-lib/ultrafeedback_binarized", split="train", streaming=True)
streamed_samples = list(islice(dataset_streamed, 3000))
dataset = Dataset.from_list(streamed_samples)

# Configure training arguments
training_args = DPOConfig(
    output_dir="gpt2-mini-dpo-output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=1,
    logging_steps=100,
    save_strategy="no",
    padding_value=tokenizer.pad_token_id
)

# Initialize the DPO trainer
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer
)

# Train the model and save metrics
train_result = trainer.train()
metrics = train_result.metrics
trainer.save_model(training_args.output_dir)
trainer.save_metrics("train", metrics)

# Optionally save metrics to JSON
with open(f"{training_args.output_dir}/train_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
