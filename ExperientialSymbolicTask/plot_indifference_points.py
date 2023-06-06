import sys

sys.path.append('..')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import torch
import argparse
from models import JointBinomialRegression
from torch.distributions import Bernoulli
import statsmodels.api as sm
import torch.nn.functional as F

models = ["7B", "13B", "30B", "65B"]
agents = ["humans", "fitted", "raw"]
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=models)
parser.add_argument("--agent", type=str, required=True, choices=agents)
args = parser.parse_args()

if args.agent == "raw":
    ee_inputs, ee_num_B_choices = torch.load('data/model=' + args.model + '_full_data.pth')
    weights = torch.load('../last_layer_65B.pth', map_location='cpu').float()[[29896, 29906]]
    probs = F.softmax(ee_inputs @ weights.t(), dim=-1)[:, 1]
    actions = (probs > probs.median(0, keepdim=True)[0]).float()

if args.agent == 'fitted':
    logits = torch.stack([torch.load('data/loo_centaur_fixed_model=65B_fold=' + str(i) + '.pth', map_location='cpu')[3] for i in range(8)], dim=-1)
    print(logits.shape)
    splits = torch.load('data/splits.pth')
    action_mat = (logits > logits.median(0, keepdim=True)[0]).float()
    actions = torch.zeros(logits.numel())
    counter = 0
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            actions[splits[i, j]] = action_mat[i, j]

df = pd.read_csv('data/block_complete.csv')
df['sub'] = pd.factorize(df['sub'])[0]

probs1 = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
probs2 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

p_experiential = []

trial_counter = 0
for subject in range(df['sub'].max() + 1):
    df_subject = df[(df['sub'] == subject) & (df['catch'] != 1) & (df['sess'] >= 0)]
    if len(df_subject) == 216:
        df_subject_le = df_subject[df_subject['elic'] == -1]
        df_subject_es = df_subject[df_subject['elic'] == 0]
        df_subject_es['trial'] = pd.factorize(df_subject_es['trial'])[0]
        p_experiential_subject = np.zeros((len(probs1), len(probs2)))

        for trial in range(df_subject_es['trial'].max() + 1):
            df_trial_es = df_subject_es[df_subject_es['trial'] == trial]
            prob1 = df_trial_es.p1.item()
            prob2 = df_trial_es.p2.item()
            prob1_index = probs1.index(prob1)
            prob2_index = probs2.index(prob2)

            if args.agent == 'humans':
                p_experiential_subject[prob1_index, prob2_index] = (df_subject_es[(df_subject_es['p1'] == prob1) & (df_subject_es['p2'] == prob2)]['cho'].item() == 1)
            elif args.agent == 'raw' or args.agent == 'fitted':
                p_experiential_subject[prob1_index, prob2_index] = (actions[trial_counter] == 0).float()
            trial_counter += 1

        p_experiential.append(p_experiential_subject)

plt.rcParams["figure.figsize"] = (1.8,1.7)

cmap = matplotlib.cm.get_cmap('Reds')

p_experiential = np.array(p_experiential)
p_experiential_mean = p_experiential.mean(0)

indifference_points = []
for i in range(p_experiential.shape[1]):
    x = np.repeat(np.array(probs2)[None], p_experiential.shape[0], axis=0).flatten()
    y = p_experiential[:, i].flatten()
    log_reg = sm.Logit(y, sm.add_constant(x)).fit()

    indifference_points.append(-log_reg.params[0] / log_reg.params[1])

plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10),  color='grey', ls='--', alpha=0.3)
plt.hlines(y=0.5, xmin=0, xmax=1, color='grey', ls='--', alpha=0.3)
plt.scatter(probs1, indifference_points, color=cmap(0.5), s=8)
sns.despine()
plt.xlabel('E-option p(win)', size=9)
plt.ylabel('Indifference point', size=9)
ax = plt.gca()
plt.xticks([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], size=9)
plt.yticks([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], size=9)
plt.tight_layout()
plt.savefig('figures/indifference_' + args.agent + '.pdf', bbox_inches='tight')
plt.show()
