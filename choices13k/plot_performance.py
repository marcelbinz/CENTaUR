import sys

sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from tqdm import tqdm
import torch
import math
from models import BinomialRegression
from torch.distributions import Bernoulli
from scipy.stats import sem


df = pd.read_json("data/c13k_problems.json", orient='index')
df2 = pd.read_csv('data/c13k_selections.csv')
print(len(df))

# llama
unsorted_probs = torch.stack([torch.load('data/loo_llama_model=65B_fold=' + str(i) + '.pth', map_location='cpu')[1] for i in range(100)], dim=-1)
ids = torch.load('data/splits.pth')
llama_probs = torch.zeros(ids.numel())
for i in range(ids.shape[0]):
    for j in range(ids.shape[1]):
        llama_probs[ids[i, j]] = unsorted_probs[i, j]

llama_probs = llama_probs.detach().numpy()

# centaur
unsorted_probs = torch.sigmoid(torch.stack([torch.load('data/loo_centaur_model=65B_fold=' + str(i) + '.pth', map_location='cpu')[2] for i in range(100)], dim=-1))
ids = torch.load('data/splits.pth')
centaur_probs = torch.zeros(ids.numel())
for i in range(ids.shape[0]):
    for j in range(ids.shape[1]):
        centaur_probs[ids[i, j]] = unsorted_probs[i, j]

centaur_probs = centaur_probs.detach().numpy()

regret_human = []
regret_random = []
regret_llama = []
regret_centaur = []
counter = 0
for index in range(len(df)):
    if counter < 9800:
        df_row = df.iloc[index]
        df2_row = df2.iloc[index]
        if df2_row.Feedback and not df2_row.Amb:
            value_A = 0
            for item_A in df_row.A:
                value_A += item_A[1] * item_A[0]

            value_B = 0
            for item_B in df_row.B:
                value_B += item_B[1] * item_B[0]

            human_value = value_B * df2_row.bRate + value_A * (1 - df2_row.bRate)
            llama_value = value_B * llama_probs[counter] + value_A * (1 - llama_probs[counter])
            centaur_value = value_B * centaur_probs[counter] + value_A * (1 - centaur_probs[counter])
            random_value = value_B * 0.5 + value_A * 0.5

            regret_human.append(max(value_A, value_B) - human_value)
            regret_random.append(max(value_A, value_B) - random_value)
            regret_llama.append(max(value_A, value_B) - llama_value)
            regret_centaur.append(max(value_A, value_B) - centaur_value)

            counter = counter + 1

regrets = [np.array(regret_random).mean(), np.array(regret_llama).mean(), np.array(regret_centaur).mean(), np.array(regret_human).mean()]
se = [sem(regret_random), sem(regret_llama), sem(regret_centaur), sem(regret_human)]
print(regrets)
print(se)

plt.rcParams["figure.figsize"] = (1.8,2.0)

plt.bar(['Random', 'LLaMA', 'CENTaUR', 'Human'], regrets, color=['C0', 'C0', 'C1', 'C0'], alpha=0.8)
sns.despine()
plt.ylabel('Regrets', size=9)
plt.ylim(0, 2.5)
plt.xticks(rotation='vertical', size=9)
plt.tight_layout()
plt.yticks(size=9)
plt.savefig('figures/choices13k_regrets.pdf', bbox_inches='tight')
plt.show()
