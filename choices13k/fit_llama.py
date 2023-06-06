import sys

sys.path.append('..')

import argparse
import torch
from models import TemperatureBinomialRegression
from torch.distributions import Binomial

models = ["7B", "13B", "30B", "65B"]
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=models)
parser.add_argument('--foldid', type=int, default=0, help='id of the testing fold')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 111333.984375 random performance
splits = torch.load('data/splits.pth')
inputs, num_choices, num_B_choices = torch.load('data/model=' + args.model + '_full_data_choices13k.pth')

inputs = inputs.to(device)
num_choices = num_choices.to(device)
num_B_choices = num_B_choices.to(device).round() # dirty hack
splits = splits.to(device)

train_index = torch.cat((splits[:, :args.foldid], splits[:, (args.foldid+1):]), dim=-1).flatten()
test_index = splits[:, args.foldid]

inputs_train = inputs[train_index]
num_choices_train = num_choices[train_index]
num_B_choices_train = num_B_choices[train_index]

inputs_test = inputs[test_index]
num_choices_test = num_choices[test_index]
num_B_choices_test = num_B_choices[test_index]

mean = inputs_train.mean(0, keepdim=True)
std = inputs_train.std(0, keepdim=True)
inputs_train = (inputs_train - mean) / std
inputs_test = (inputs_test - mean) / std

logreg = TemperatureBinomialRegression(inputs_train.shape[1], args.model).to(device)
logreg.fit(
    inputs_train,
    num_choices_train,
    num_B_choices_train
)

probs = logreg(inputs_test)
llf_test = -Binomial(total_count=num_choices_test, probs=probs).log_prob(num_B_choices_test).sum().item()

print(llf_test)
torch.save([llf_test, probs], 'data/loo_llama_model=' + args.model + '_fold=' + str(args.foldid) + '.pth')
