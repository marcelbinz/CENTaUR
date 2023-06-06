import pandas as pd
import torch
import argparse

models = ["7B", "13B", "30B", "65B"]
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=models)
args = parser.parse_args()

num_splits = 8
num_participants = 98

inputs = torch.cat([torch.load('data/model=' + str(args.model) + '_participant=' + str(participant) + '.pth')[0] for participant in range(num_participants)], dim=0)
targets = torch.cat([torch.load('data/model=' + str(args.model) + '_participant=' + str(participant) + '.pth')[1] for participant in range(num_participants)], dim=0)

print(inputs.shape)
print(targets.shape)

splits = torch.randperm(inputs.shape[0]).reshape(-1, num_splits)
torch.save(splits, 'data/splits.pth')

torch.save([inputs, targets], 'data/model=' + args.model + '_full_data.pth')
