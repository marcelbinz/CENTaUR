import torch
import pandas as pd
import argparse

models = ["7B", "13B", "30B", "65B"]
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=models)
parser.add_argument('--recompute-splits', action='store_true', default=False, help='whether to recompute splits')
args = parser.parse_args()

llama_features = torch.load('data/model=' + args.model + '.pth')[0]

inputs = []
num_choices = []
num_B_choices = []

num_splits = 100

df = pd.read_csv('data/c13k_selections.csv')

for index in range(len(df)):
    df_row = df.iloc[index]
    if df_row.Feedback and not df_row.Amb:
        inputs.append(llama_features[[index]])
        num_choices.append(df_row.n * 5)
        num_B_choices.append(df_row.n * 5 * df_row.bRate)

inputs = torch.cat(inputs, dim=0)[:9800]
num_choices = torch.Tensor(num_choices)[:9800]
num_B_choices = torch.Tensor(num_B_choices)[:9800]

torch.save([inputs, num_choices, num_B_choices], 'data/model=' + args.model + '_full_data_choices13k.pth')

if args.recompute_splits:
    splits = torch.randperm(inputs.shape[0]).reshape(-1, num_splits)
    torch.save(splits, 'data/splits.pth')
