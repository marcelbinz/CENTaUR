import glob
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from tqdm import tqdm
import torch
import argparse
import torch.nn.functional as F
from torch.distributions import Bernoulli

conditions = ["equal", "unequal"]
agents = ["human", "raw", "fitted"]
parser = argparse.ArgumentParser()
parser.add_argument("--agent", type=str, required=True, choices=agents)
parser.add_argument("--condition", type=str, required=True, choices=conditions)
args = parser.parse_args()

if args.agent == 'raw':
    probs = torch.stack([torch.load('data/loo_llama_fixed_model=65B_fold=' + str(i) + '.pth', map_location='cpu')[1] for i in range(100)], dim=-1)
    splits = torch.load('data/splits.pth')
    action_mat = Bernoulli(probs=probs).sample()
    actions = torch.zeros(probs.numel())
    counter = 0
    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            actions[splits[i, j]] = action_mat[i, j]


if args.agent == 'fitted':
    logits = torch.stack([torch.load('data/loo_centaur_fixed_model=65B_fold=' + str(i) + '.pth', map_location='cpu')[2] for i in range(100)], dim=-1)
    splits = torch.load('data/splits.pth')
    action_mat = Bernoulli(logits=logits).sample()
    actions = torch.zeros(logits.numel())
    counter = 0
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            actions[splits[i, j]] = action_mat[i, j]

# preprocess data
df = pd.read_csv('data/exp1.csv')
forced_choices_1 = np.zeros(len(df))
choices_raw = np.zeros(len(df))
choices_fitted = np.zeros(len(df))
num_participants = df.participant.max() + 1
num_tasks = df.task.max() + 1
row_counter1 = 0
row_counter2 = 0
for participant in tqdm(range(num_participants)):
    df_participant = df[df['participant'] == participant]
    for task in range(num_tasks):
        df_task = df_participant[df_participant['task'] == task]
        choices_1 = df_task[df_task.trial < 4].choice.sum()
        num_trials = df_task.trial.max() + 1
        row_counter1 = row_counter1 + 4
        for trial in range(4, num_trials):
            forced_choices_1[row_counter1] = choices_1
            if args.agent == 'raw':
                choices_raw[row_counter1] = actions[row_counter2].item()
            if args.agent == 'fitted':
                choices_fitted[row_counter1] = actions[row_counter2].item()
            row_counter1 = row_counter1 + 1
            row_counter2 = row_counter2 + 1

df['forced_choices_1'] = forced_choices_1.astype('int')
if args.agent == 'raw':
    df['choice'] = choices_raw
if args.agent == 'fitted':
    df['choice'] = choices_fitted
if args.condition == 'equal':
    # extract data
    df = df[(df.trial == 4) & (df.forced_choices_1 == 2)]
    reward_differences = (df.expected_reward0 - df.expected_reward1).to_numpy().astype(float)
    choices = 1 - df.choice.to_numpy()
    horizon = (df[(df.trial == 4) & (df.forced_choices_1 == 2)].horizon == 10).to_numpy().astype(float)
    interaction = horizon * reward_differences

    log_reg = sm.Logit(choices, np.stack((reward_differences, horizon, interaction, np.ones(reward_differences.shape)), axis=-1)).fit()

elif args.condition == 'unequal':
    # case: x3 1
    df_31 = df[(df.trial == 4) & (df.forced_choices_1 == 3)]
    reward_differences_31 = (df_31.expected_reward0 - df_31.expected_reward1).to_numpy().astype(float)
    choices_31 = 1 - df_31.choice.to_numpy()
    horizons_31 = (df[(df.trial == 4) & (df.forced_choices_1 == 3)].horizon == 10).to_numpy().astype(float)

    # case: x3 0
    df_13 = df[(df.trial == 4) & (df.forced_choices_1 == 1)]
    reward_differences_13 = (df_13.expected_reward1 - df_13.expected_reward0).to_numpy().astype(float)
    choices_13 = df_13.choice.to_numpy()
    horizons_13 = (df[(df.trial == 4) & (df.forced_choices_1 == 1)].horizon == 10).to_numpy().astype(float)

    choices = np.concatenate((choices_31, choices_13), axis=0)
    reward_differences = np.concatenate((reward_differences_31, reward_differences_13), axis=0)
    horizon = np.concatenate((horizons_31, horizons_13), axis=0)
    interaction = horizon * reward_differences

    log_reg = sm.Logit(choices, np.stack((reward_differences, horizon, interaction, np.ones(reward_differences.shape)), axis=-1)).fit()

x_reward_differences = np.linspace(-30, 30, 1000)
x_horizon6 = np.ones(1000)
x_6 = np.stack((x_reward_differences, x_horizon6, x_horizon6 * x_reward_differences, np.ones(1000)), axis=-1)
y_6 = log_reg.predict(x_6)

x_reward_differences = np.linspace(-30, 30, 1000)
x_horizon1 = np.zeros(1000)
x_1 = np.stack((x_reward_differences, x_horizon1, x_horizon1 * x_reward_differences, np.ones(1000)), axis=-1)
y_1 = log_reg.predict(x_1)

# make plot
plt.rcParams["figure.figsize"] = (1.8,2.0)
plt.plot(x_1[:, 0], y_1, color='C0' if args.condition == 'equal' else 'C1')
plt.plot(x_6[:, 0], y_6, color='C0' if args.condition == 'equal' else 'C1', ls='--')
sns.despine()
plt.xlabel('Reward difference', size=9)
if args.condition == 'equal':
    plt.ylabel('p(first option)', size=9)
else:
    plt.ylabel('p(more informative)', size=9)

plt.legend(['Horizon 1',  'Horizon 6',], frameon=False, bbox_to_anchor=(0.0, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, handletextpad=0.5, columnspacing=0.6, ncol=1, prop={'size': 9})
plt.ylim(0, 1)
plt.xlim(-30, 30)
plt.yticks(size=9)
plt.xticks(size=9)
plt.tight_layout()
plt.savefig('figures/probs_agent=' + args.agent + '_condition=' + args.condition + '.pdf', bbox_inches='tight')
plt.show()
