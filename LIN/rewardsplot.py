import json
import pandas as pd
import matplotlib.pyplot as plt


log_path = 'dpo_negative_output/dpo_log_history.json' 
with open(log_path, 'r', encoding='utf-8') as f:
    history = json.load(f)


df = pd.DataFrame(history)


x = df['rewards/margins']   
y = df['rewards/chosen']     
plt.figure()
plt.scatter(x, y, alpha=0.7)
plt.xlabel('KL Divergence')
plt.ylabel('Reward (chosen)')
plt.title('Reward vs. KL Divergence (scatter)')
plt.grid(True)
plt.tight_layout()
plt.savefig('kl_scatter.png')
plt.show()


df['epoch_int'] = df['epoch'].astype(int)
mean_reward = df.groupby('epoch_int')['rewards/chosen'].mean()

plt.figure()
plt.plot(mean_reward.index, mean_reward.values, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Mean Reward (chosen)')
plt.title('Mean Reward vs. Epoch (integer epochs)')
plt.grid(True)
plt.tight_layout()
plt.savefig('mean_reward_int_epoch.png')
plt.show()
