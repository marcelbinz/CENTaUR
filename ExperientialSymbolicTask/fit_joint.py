import sys

sys.path.append('..')

import pandas as pd
import numpy as np
import torch
import argparse
from models import JointBinomialRegression
from torch.distributions import Binomial
import statsmodels.api as sm

models = ["7B", "13B", "30B", "65B"]
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=models)
parser.add_argument('--foldid', type=int, default=0, help='id of the testing fold')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data
ee_inputs, ee_num_B_choices = torch.load('data/model=' + args.model + '_full_data.pth')
dfd_inputs, dfd_num_choices, dfd_num_B_choices = torch.load('../choices13k/data/model=' + args.model + '_full_data_choices13k.pth')
ht_inputs, ht_num_B_choices, _ = torch.load('../HorizonTask/data/model=' + args.model + '_full_data.pth')

# everything to cuda if needed
ee_inputs = ee_inputs.to(device)
dfd_inputs = dfd_inputs.to(device)
ht_inputs = ht_inputs.to(device)
ee_num_B_choices = ee_num_B_choices.to(device)
ee_num_choices = torch.ones_like(ee_num_B_choices).to(device)
dfd_num_B_choices = dfd_num_B_choices.round().to(device)
dfd_num_choices = dfd_num_choices.to(device)
ht_num_B_choices = ht_num_B_choices.to(device)
ht_num_choices = torch.ones_like(ht_num_B_choices).to(device)

# normalize on training data
train_inputs = torch.cat((dfd_inputs, ht_inputs), dim=0)
mean = train_inputs.mean(0, keepdim=True)
std = train_inputs.std(0, keepdim=True)
dfd_inputs = (dfd_inputs - mean) / std
ht_inputs = (ht_inputs - mean) / std
ee_inputs = (ee_inputs - mean) / std

# extra val and test data
splits = torch.load('data/splits.pth').to(device)
val_index = torch.cat((splits[:, :args.foldid], splits[:, (args.foldid+1):]), dim=-1).flatten()
test_index = splits[:, args.foldid]

inputs_validation = ee_inputs[val_index]
num_choices_validation = ee_num_choices[val_index].to(device)
num_B_choices_validation = ee_num_B_choices[val_index].to(device)

inputs_test = ee_inputs[test_index]
num_choices_test = ee_num_choices[test_index].to(device)
num_B_choices_test = ee_num_B_choices[test_index].to(device)

alphas = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
temps = np.linspace(0.05, 1.0, 20)

loo_validation = torch.zeros(len(alphas), len(temps))

for i, alpha in enumerate(alphas):
    print(i)
    for k, temp in enumerate(temps):
        # fit model
        logreg = JointBinomialRegression(train_inputs.shape[1], alpha=alpha, temp=temp).to(device)
        logreg.fit(
            dfd_inputs,
            dfd_num_choices,
            dfd_num_B_choices,
            ht_inputs,
            ht_num_choices,
            ht_num_B_choices
        )
        loo_validation[i, k] = -Binomial(total_count=num_choices_validation, logits=logreg(inputs_validation)).log_prob(num_B_choices_validation).mean().item()

# refitting with best indices
best_indices = (loo_validation==torch.min(loo_validation)).nonzero()
best_alpha = alphas[best_indices[0, 0]]
best_temp = temps[best_indices[0, 1]]
logreg = JointBinomialRegression(train_inputs.shape[1], alpha=best_alpha, temp=best_temp).to(device)
logreg.fit(
    dfd_inputs,
    dfd_num_choices,
    dfd_num_B_choices,
    ht_inputs,
    ht_num_choices,
    ht_num_B_choices
)

logits = logreg(inputs_test)
llf_test = -Binomial(total_count=num_choices_test, logits=logits).log_prob(num_B_choices_test).sum().item()
print(llf_test)
torch.save([llf_test, best_alpha, best_temp, logits], 'data/loo_centaur_fixed_model=' + args.model + '_fold=' + str(args.foldid) + '.pth')
