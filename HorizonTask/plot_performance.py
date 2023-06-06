import sys

sys.path.append('..')

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from tqdm import tqdm
import torch
import math
import argparse
from torch.distributions import Binomial
from scipy.stats import sem

df = pd.read_csv('data/exp1.csv')
df = df[df.trial >= 4]

# human regret
r_max = df[["expected_reward0", "expected_reward1"]].max(axis=1)
r_obs = df.reward
print(len(r_obs))
human_regret = (r_max - r_obs).to_numpy().mean()
human_regret_se = sem((r_max - r_obs).to_numpy())

# random regret
r_random = 0.5 * df["expected_reward0"] +  0.5 * df["expected_reward1"]
random_regret = (r_max - r_random).to_numpy().mean()
random_regret_se = sem((r_max - r_random).to_numpy())

# raw regret
unsorted_probs = torch.stack([torch.load('data/loo_llama_fixed_model=65B_fold=' + str(i) + '.pth', map_location='cpu')[1] for i in range(100)], dim=-1)
print(unsorted_probs.shape)
ids = torch.load('data/splits.pth')
probs = torch.zeros(ids.numel())
for i in range(ids.shape[0]):
    for j in range(ids.shape[1]):
        probs[ids[i, j]] = unsorted_probs[i, j]

probs = probs[:len(df)].detach().numpy()
r_raw = (1-probs) * df["expected_reward0"] + probs * df["expected_reward1"]
raw_regret = (r_max - r_raw).to_numpy().mean()
raw_regret_se = sem((r_max - r_raw).to_numpy())

# fitted regret
unsorted_probs = torch.sigmoid(torch.stack([torch.load('data/loo_centaur_fixed_model=65B_fold=' + str(i) + '.pth', map_location='cpu')[2] for i in range(100)], dim=-1))
print(unsorted_probs.shape)
ids = torch.load('data/splits.pth')
probs = torch.zeros(ids.numel())
for i in range(ids.shape[0]):
    for j in range(ids.shape[1]):
        probs[ids[i, j]] = unsorted_probs[i, j]

probs = probs[:len(df)].detach().numpy()
r_fitted = (1-probs) * df["expected_reward0"] + probs * df["expected_reward1"]
fitted_regret = (r_max - r_fitted).to_numpy().mean()
fitted_regret_se = sem((r_max - r_fitted).to_numpy())

regrets = [random_regret, raw_regret, fitted_regret, human_regret]
print(random_regret)
print(human_regret)
print(raw_regret)
print(fitted_regret)

print(random_regret_se)
print(human_regret_se)
print(raw_regret_se)
print(fitted_regret_se)

plt.rcParams["figure.figsize"] = (1.8,2.0)

plt.bar(['Random', 'LLaMA', 'CENTaUR', 'Human'], regrets, color=['C0', 'C0', 'C1', 'C0'], alpha=0.8)
sns.despine()
plt.ylabel('Regrets', size=9)
plt.ylim(0, 10)
plt.xticks(rotation='vertical', size=9)
plt.tight_layout()
plt.yticks(size=9)
plt.savefig('figures/horizon_regrets.pdf', bbox_inches='tight')
plt.show()
