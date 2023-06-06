import argparse
import torch
import torch.nn.functional as F
from torch.distributions import Binomial
from tqdm import tqdm

beast_pred, num_choices, num_B_choices = torch.load('data/model=BEAST_full_data_choices13k.pth')
num_B_choices = num_B_choices.round()
splits = torch.load('data/splits.pth')
num_splits = splits.shape[1]

errors = torch.linspace(0, 1, 101)
llf_per_fold = torch.zeros(num_splits)

for fold in tqdm(range(num_splits)):
    train_index = torch.cat((splits[:, :fold], splits[:, (fold+1):]), dim=-1).flatten()
    test_index = splits[:, fold]

    beast_pred_train = beast_pred[train_index]
    num_choices_train = num_choices[train_index]
    num_B_choices_train = num_B_choices[train_index]

    beast_pred_test = beast_pred[test_index]
    num_choices_test = num_choices[test_index]
    num_B_choices_test = num_B_choices[test_index]

    llf_per_error = torch.zeros(len(errors))
    for i, error in enumerate(errors):
        beast_pred_train_error = (1-error) * beast_pred_train + error * 0.5
        llf_per_error[i] = -Binomial(total_count=num_choices_train, probs=beast_pred_train_error).log_prob(num_B_choices_train).sum().item()

    best_error = errors[torch.argmin(llf_per_error).item()]
    beast_pred_test_error = (1-best_error) * beast_pred_test + best_error * 0.5
    llf_per_fold[fold] = -Binomial(total_count=num_choices_test, probs=beast_pred_test_error).log_prob(num_B_choices_test).sum().item()

print(llf_per_fold.sum())
torch.save(llf_per_fold.sum(), 'data/loo_beast.pth')
