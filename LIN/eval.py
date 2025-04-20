import os
import json
import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, pipeline
from tqdm.auto import tqdm
import numpy as np
import openai

# ?? API key(?????????,??????)
openai.api_key = os.getenv("OPENAI_API_KEY") or "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"  

# 1. ???????
FT_MODEL_DIR = "dpo_negative_output/final_model"  # ?????????
BASELINE_MODEL = "gpt2"                           # Baseline ???? GPT-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2. ????? tokenizer
ft_tokenizer = GPT2TokenizerFast.from_pretrained(FT_MODEL_DIR)
ft_model = GPT2LMHeadModel.from_pretrained(FT_MODEL_DIR).to(DEVICE)
ft_model.eval()

base_tokenizer = GPT2TokenizerFast.from_pretrained(BASELINE_MODEL)
base_model = GPT2LMHeadModel.from_pretrained(BASELINE_MODEL).to(DEVICE)
base_model.eval()

# 3. ??????
test_ds = load_dataset("imdb", split="test").shuffle(seed=42).select(range(100))
prompts = ["Review: " + txt[:200] + "\nContinue negatively:\n" for txt in test_ds["text"]]

# ????????
results = {}

# 4. Perplexity
def compute_avg_perplexity(model, tokenizer, texts):
    perps = []
    for txt in tqdm(texts, desc="Perplexity"):
        enc = tokenizer(txt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**enc, labels=enc.input_ids)
        perps.append(torch.exp(outputs.loss).item())
    return float(np.mean(perps))

results["fine_tuned_perplexity"] = compute_avg_perplexity(ft_model, ft_tokenizer, prompts)
results["baseline_perplexity"]   = compute_avg_perplexity(base_model, base_tokenizer, prompts)

# 5. Win Rate(? GPT-4 ???????)
def compare_with_gpt4(prompt, resp_ft, resp_base):
    messages = [
        {"role": "system", "content": "You are a movie review expert. Decide which response is more negative. Reply with 'FT' or 'BASE'."},
        {"role": "user", "content": f"Prompt:\n{prompt}\n\nFine-tuned Response:\n{resp_ft}\n\nBaseline Response:\n{resp_base}\n\nWhich is more negative?"}
    ]
    r = openai.ChatCompletion.create(model="gpt-4", messages=messages, temperature=0.0)
    return r.choices[0].message.content.strip()

win = 0
compare_count = 20  # ?? ????? 20 ?? GPT-4 ??,??????
for i, p in enumerate(tqdm(prompts[:compare_count], desc="WinRate")):
    # Fine-tuned response
    in_ft = ft_tokenizer(p, return_tensors="pt").to(DEVICE)
    out_ft = ft_model.generate(**in_ft, max_new_tokens=50)
    resp_ft = ft_tokenizer.decode(out_ft[0][in_ft.input_ids.shape[-1]:], skip_special_tokens=True)

    # Baseline response
    in_bs = base_tokenizer(p, return_tensors="pt").to(DEVICE)
    out_bs = base_model.generate(**in_bs, max_new_tokens=50)
    resp_bs = base_tokenizer.decode(out_bs[0][in_bs.input_ids.shape[-1]:], skip_special_tokens=True)

    if compare_with_gpt4(p, resp_ft, resp_bs) == "FT":
        win += 1
results["win_rate"] = win / compare_count

# 6. Sentiment Accuracy(?????????)
sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=-1)
correct = 0
for p in tqdm(prompts, desc="SentimentAcc"):
    in_ft = ft_tokenizer(p, return_tensors="pt").to(DEVICE)
    out_ft = ft_model.generate(**in_ft, max_new_tokens=50)
    resp_ft = ft_tokenizer.decode(out_ft[0][in_ft.input_ids.shape[-1]:], skip_special_tokens=True)
    pred = sentiment(resp_ft)[0]["label"]
    if pred in ["1 star", "2 stars"]:
        correct += 1
results["sentiment_accuracy"] = correct / len(prompts)

# 7. Distinct-1(?????)
all_texts = []
for p in prompts:
    in_ft = ft_tokenizer(p, return_tensors="pt").to(DEVICE)
    out_ft = ft_model.generate(**in_ft, max_new_tokens=50)
    text = ft_tokenizer.decode(out_ft[0][in_ft.input_ids.shape[-1]:], skip_special_tokens=True)
    all_texts.append(text)
tokens = [tok for txt in all_texts for tok in txt.split()]
results["distinct_1"] = len(set(tokens)) / len(tokens)

# 8. ????? JSON ??
with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("All evaluation metrics saved to evaluation_results.json")
