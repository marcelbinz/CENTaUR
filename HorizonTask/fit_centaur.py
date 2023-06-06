import sys

sys.path.append('..')

import argparse
import torch
from models import BinomialRegression
from torch.distributions import Binomial

models = ["7B", "13B", "30B", "65B"]
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=models)
parser.add_argument('--foldid', type=int, default=0, help='id of the testing fold')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# XXXX random performance
splits = torch.load('data/splits.pth')
inputs, num_B_choices, _ = torch.load('data/model=' + args.model + '_full_data.pth')

inputs = inputs.to(device)
num_choices = torch.ones_like(num_B_choices).to(device)
num_B_choices = num_B_choices.to(device)
splits = splits.to(device)

# get splits
train_index = torch.cat((splits[:, :args.foldid], splits[:, (args.foldid+1):]), dim=-1).flatten()
test_index = splits[:, args.foldid]

# cross-validation for regularization parameter
alphas = [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
num_inner_folds = 11
train_index = train_index.reshape(-1, num_inner_folds)
llf_train = torch.zeros(len(alphas), num_inner_folds)
llf_validation = torch.zeros(len(alphas), num_inner_folds)

for fold in range(num_inner_folds):
    print(fold, flush=True)
    inner_training_index = torch.cat((train_index[:, :fold], train_index[:, (fold+1):]), dim=-1).flatten()
    inner_validation_index = train_index[:, fold]

    # extract data
    inputs_train = inputs[inner_training_index]
    num_choices_train = num_choices[inner_training_index]
    num_B_choices_train = num_B_choices[inner_training_index]

    inputs_validation = inputs[inner_validation_index]
    num_choices_validation = num_choices[inner_validation_index]
    num_B_choices_validation = num_B_choices[inner_validation_index]

    inputs_test = inputs[test_index]
    num_choices_test = num_choices[test_index]
    num_B_choices_test = num_B_choices[test_index]

    # normalize data
    mean = inputs_train.mean(0, keepdim=True)
    std = inputs_train.std(0, keepdim=True)
    inputs_train = (inputs_train - mean) / std
    inputs_validation = (inputs_validation - mean) / std
    inputs_test = (inputs_test - mean) / std

    for alpha_index, alpha in enumerate(alphas):
        # fit model
        logreg = BinomialRegression(inputs_train.shape[1], alpha=alpha).to(device)
        logreg.fit(
            inputs_train,
            num_choices_train,
            num_B_choices_train
        )

        llf_train[alpha_index, fold] = -Binomial(total_count=num_choices_train, logits=logreg(inputs_train)).log_prob(num_B_choices_train).mean().item()
        llf_validation[alpha_index, fold] = -Binomial(total_count=num_choices_validation, logits=logreg(inputs_validation)).log_prob(num_B_choices_validation).mean().item()

# get best alpha
best_alpha_index = llf_validation.sum(-1).argmin()
best_alpha = alphas[best_alpha_index]

# refitting on all training data
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

logreg = BinomialRegression(inputs_train.shape[1], alpha=best_alpha).to(device)
logreg.fit(
    inputs_train,
    num_choices_train,
    num_B_choices_train
)

logits = logreg(inputs_test)
llf_test = -Binomial(total_count=num_choices_test, logits=logits).log_prob(num_B_choices_test).sum().item()

print(llf_test)
torch.save([llf_test, best_alpha, logits], 'data/loo_centaur_fixed_model=' + args.model + '_fold=' + str(args.foldid) + '.pth')
