import pandas as pd
import torch
import argparse

models = ["7B", "13B", "30B", "65B"]
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=models)
parser.add_argument('--recompute-splits', action='store_true', default=False, help='whether to recompute splits')
args = parser.parse_args()

# which experiments to use
exps = ['exp1', 'exp2']

inputs = []
targets = []
ids = []

participant_counter = 0
num_trials_per_participant = 1120 # TODO careful with this
num_splits = 100

for exp in exps:
    df = pd.read_csv('data/' + exp + '.csv')
    num_tasks = df.task.max() + 1
    num_participants = df.participant.max() + 1

    for participant in range(num_participants):
        df_participant = df[(df['participant'] == participant)]
        for task in range(num_tasks):
            # extract llama
            llama_features, human_actions = torch.load('data/model=' + args.model + '_data=' + exp + '_participant=' + str(participant) + '_task=' + str(task) + '.pth')
            inputs.append(llama_features[4:])
            targets.append(human_actions[4:])

        ids.append(participant_counter * torch.ones(num_trials_per_participant))
        participant_counter = participant_counter + 1

inputs = torch.cat(inputs, dim=0)
targets = torch.cat(targets, dim=0)
ids = torch.cat(ids, dim=0).long()
torch.save([inputs, targets, ids], 'data/model=' + args.model + '_full_data.pth')

if args.recompute_splits:
    splits = torch.randperm(inputs.shape[0]).reshape(-1, num_splits)
    torch.save(splits, 'data/splits.pth')

print('done', flush=True)
