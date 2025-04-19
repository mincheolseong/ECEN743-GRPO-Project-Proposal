# run_ppo.py
print(">>> Start script...ff", flush=True)
import warnings
import re
import random
print(">>> Imported warnings", flush=True)
import torch
print(">>> Imported torch", flush=True)
from transformers import GPT2LMHeadModel, GPT2Tokenizer
print(">>> Imported transformers", flush=True)
from datasets import load_from_disk
print(">>> Imported datasets", flush=True)
from trl import PPOTrainer, PPOConfig
print(">>> Imported PPOTrainer", flush=True)
from trl import AutoModelForCausalLMWithValueHead
print(">>> Imported ValueHead", flush=True)
from trl.core import LengthSampler
from transformers import pipeline
import csv

warnings.filterwarnings("ignore", message="`resume_download` is deprecated")
warnings.filterwarnings("ignore", message="Xformers is not installed correctly")
warnings.filterwarnings("ignore", message="No dataset is provided.")

n_epochs = 20
n_samples = 1

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model
print(">>> Loading tokenizer and model...", flush=True)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2").to(device)


# Load preprocessed IMDb data (negative reviews only)
dataset = load_from_disk("tokenized_imdb_negative")

# Sample a few prompts for training
prompt_templates = [
    "This movie is so bad I had to leave. Continue the review:\n",
    "This film is a waste of time. Finish this:\n",
    "I hated everything about this movie. Explain why:\n",
    "The worst film ever. Expand the comment:\n",
    "The story was painfully boring. Go on:\n",
    "The direction and acting were terrible. Elaborate:\n",
]

#prompts = [random.choice(prompt_templates) + "\n" + tokenizer.decode(example["input_ids"][:64]) # the number of tokens 
#           for example in dataset.select(range(50))] # 50 for minimal experience, up to 12,500
           
prompts = [
    random.choice(prompt_templates)
    for _ in range(50)
]           
           
#print("prompts", prompts)

# Load reward model (IMDb classifier)
print(">>> Loading reward model...", flush=True)
#sentiment_pipe = pipeline(
#    "text-classification",
#    model="textattack/roberta-base-imdb",
#    device=0 if torch.cuda.is_available() else -1
#)
#
#toxicity_pipe = pipeline(
#    "text-classification",
#    model="unitary/toxic-bert",
#    device=0 if torch.cuda.is_available() else -1
#)
print(">>> Starting training...", flush=True)

# PPO config
ppo_config = PPOConfig(
    model_name="gpt2",
    learning_rate=1e-5,
    batch_size=8,
    mini_batch_size=2,
    ppo_epochs=4,
    log_with="tensorboard",  
    kl_penalty="kl",         
    target_kl=0.2,
    ratio_threshold=20.0,
    early_stopping=True
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer
)

log_file = open("ppo_logs/ppo_training_log.csv", "w", newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(["epoch", "step", "reward", "kl_divergence", "GPT-2 response", "PPO response"])

neg_patterns = [
    r"\bbad\b", r"\bterrible\b", r"\bawful\b", r"\bhorrible\b", r"\bpoor\b", r"\bboring\b", r"\bslow\b", r"\bdull\b",
    r"\bdisappointing\b", r"\bannoying\b", r"waste of time", r"not worth it", r"\bunbearable\b", r"\bmediocre\b",
    r"\bforgettable\b", r"\bflawed\b", r"\bunwatchable\b", r"\bgarbage\b", r"\btrash\b", r"\bmess\b", r"\bcheesy\b",
    r"\bcringe\b", r"\bregret\b", r"\bpathetic\b", r"\bsucks\b", r"\bstupid\b", r"\bnonsense\b", r"makes no sense",
    r"didn't like", r"couldn't finish", r"hated it", r"\bconfusing\b", r"\bpredictable\b", r"\bbuggy\b",
    r"\bridiculous\b", r"\babsurd\b", r"\boverrated\b", r"\bunderrated\b", r"\bincoherent\b", r"\bpainful\b",
    r"\bfake\b", r"\bpointless\b", r"\brepetitive\b", r"\bshallow\b", r"\bcliched\b", r"\blame\b", r"\blazy\b",
    r"\bbroken\b", r"poorly made", r"script was bad", r"bad acting", r"bad writing", r"plot holes",
    r"no plot", r"no development", r"no character arc", r"too long", r"dragged", r"drawn out",
    r"overacted", r"underacted", r"low budget", r"\bcheap\b", r"low quality", r"poor direction",
    r"\binconsistent\b", r"\bunbelievable\b", r"\bforced\b", r"bad pacing", r"terrible ending",
    r"no logic", r"makes you sleep", r"predictable twists", r"hate the ending", r"poor performance",
    r"fails to deliver", r"didn't work", r"had issues", r"not engaging", r"hard to watch", r"not funny",
    r"not scary", r"not interesting", r"annoying characters", r"\boverdone\b", r"\bpretentious\b",
    r"\bwannabe\b", r"\boveredited\b", r"\bunderwhelming\b", r"\bdisconnected\b", r"badly shot"
]


def repetition_reward(text: str) -> float:
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)  # distinct-1 


def length_reward(text: str, min_len=10, max_len=64) -> float:
    length = len(text.split())
    return min(1.0, max(0.0, (length - min_len) / (max_len - min_len)))


def negativity_reward(text: str) -> float:
    text_lower = text.lower()

    match_count = sum(bool(re.search(pat, text_lower)) for pat in neg_patterns)
    return min(1.0, match_count / 2.0)


def combo_reward(text: str, w_rep=0.05, w_len=0.05, w_neg=0.9) -> float:
    r1 = repetition_reward(text)
    r2 = length_reward(text)
    r3 = negativity_reward(text)
    total = w_rep * r1 + w_len * r2 + w_neg * r3
    return min(1.0, total)

all_queries = []
all_responses = []
all_rewards = []

# Training loop
for epoch in range(n_epochs):
    print(f"\n=== Epoch {epoch+1}/{n_epochs} ===")
    
    for step, prompt in enumerate(prompts): # epoch -> step: naming issue
        rewards = []
        responses = []

        for sample_idx in range(n_samples):  # ***
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device).long()
            
            with torch.no_grad():
                baseline_output = model.pretrained_model.generate(  # ***
                    input_ids,
                    max_new_tokens=80,
                    min_new_tokens=20,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
                gpt2_response = tokenizer.decode(  # ***
                    baseline_output[0][input_ids.shape[-1]:],
                    skip_special_tokens=True
                )            

            generation_output = model.generate(
                input_ids,
                max_new_tokens=80,
                min_new_tokens=10,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                eos_token_id=None
            )
            
            if generation_output.shape[-1] <= input_ids.shape[-1]:
                print("Empty generation. Skipping.")
                continue
                
            response = tokenizer.decode(generation_output[0][input_ids.shape[-1]:], skip_special_tokens=True)
            
            if len(response.strip()) == 0:
                print("Empty string response. Skipping.")
                continue
                
            if "<a href=" in response or "http" in response:
                print("URL-like response. Skipping.")
                continue
        
            responses.append(response)

            #reward_output = reward_pipe(response)
            #reward_score = reward_output[0]["score"]
            reward_score = combo_reward(response)
            rewards.append(torch.tensor(reward_score).to(device))

            # log single sample
            print(f"[Epoch {epoch+1} | Step {step+1}/{len(prompts)} | "
                  f"Reward: {reward_score:.4f} | Response: {response[:80]}...", flush=True)  # 
                        
        if len(rewards) == 0:
            print(f"[Epoch {epoch+1} | Step {step+1}] Skipped: No valid response.")
            continue  # skip this step
    
        avg_reward = torch.mean(torch.stack(rewards))  # 
        best_response = responses[rewards.index(max(rewards))]  # 
        
        
        query_tensor = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        query_tensor = {k: v.to(device) for k, v in query_tensor.items()}
        query_tensor["input_ids"] = query_tensor["input_ids"].long()

        response_tensor = tokenizer(best_response, return_tensors="pt", padding=True, truncation=True)
        response_tensor = {k: v.to(device) for k, v in response_tensor.items()}
        response_tensor["input_ids"] = response_tensor["input_ids"].long()

        all_queries.append(query_tensor["input_ids"].squeeze(0))
        all_responses.append(response_tensor["input_ids"].squeeze(0))
        all_rewards.append(avg_reward)

        
        print(f"Prompt[:100]: {prompt[:100]}")  # 
        print(f"Response[:100]: {response[:100]}")  # 
        
        #decoded_query = tokenizer.decode(query_tensor.tolist(), skip_special_tokens=True)
        #decoded_response = tokenizer.decode(response_tensor.tolist(), skip_special_tokens=True)
        decoded_query = tokenizer.decode(query_tensor["input_ids"].squeeze(0).tolist(), skip_special_tokens=True)
        decoded_response = tokenizer.decode(response_tensor["input_ids"].squeeze(0).tolist(), skip_special_tokens=True)

        print(f"Decoded query[:100]: {decoded_query[:100]}")
        print(f"Decoded PPO target response[:100]: {decoded_response[:100]}")
        
        r1 = repetition_reward(response)
        r2 = length_reward(response)
        r3 = negativity_reward(response)
        combo = combo_reward(response)
        print(f"Reward components -> repetition: {r1:.2f}, length: {r2:.2f}, negativity: {r3:.2f}, combo: {combo:.2f}")

        if len(all_queries) == ppo_config.batch_size:
            train_stats = ppo_trainer.step(all_queries, all_responses, all_rewards)
            all_queries, all_responses, all_rewards = [], [], []

            csv_writer.writerow([
                epoch + 1, step + 1,
                avg_reward.item(),
                train_stats.get("kl", train_stats.get("objective/kl", None)),
                gpt2_response,
                best_response
            ])
        #train_stats = ppo_trainer.step([query_tensor], [response_tensor], [avg_reward])  # 
        #kl_value = train_stats.get("kl", train_stats.get("objective/kl", None))
        #kl_value = train_stats.get("kl", train_stats.get("objective/kl", None)) if 'train_stats' in locals() else None

        

        #csv_writer.writerow([epoch + 1, step + 1, avg_reward.item(), kl_value, gpt2_response, best_response])  # 
        #csv_writer.writerow([epoch + 1, step + 1, avg_reward.item(), kl_value, gpt2_response, best_response])
if all_queries:
    train_stats = ppo_trainer.step(all_queries, all_responses, all_rewards)


print("Training complete.")

# Save fine-tuned model
model.save_pretrained("ppo_gpt2_finetuned_model")
tokenizer.save_pretrained("ppo_gpt2_finetuned_model")

print("Saving complete.")

