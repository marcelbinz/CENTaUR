import pandas as pd
import torch

# which experiments to use
exps = ['exp1', 'exp2']

inputs_hybrid = []
targets_hybrid = []
ids = []

participant_counter = 0
num_trials_per_participant = 1120 # TODO careful with this

for exp in exps:
    df = pd.read_csv('data/' + exp + '.csv')
    num_tasks = df.task.max() + 1
    num_participants = df.participant.max() + 1

    for participant in range(num_participants):
        df_participant = df[(df['participant'] == participant)]
        for task in range(num_tasks):
            # extract hybrid
            df_task = df_participant[(df_participant['task'] == task)]
            num_trials = df_task.trial.max() + 1

            hybrid_features = torch.zeros(num_trials, 3)
            hybrid_actions = torch.zeros(num_trials)

            m = torch.Tensor([50.0, 50.0])
            s = torch.Tensor([400.0, 400.0])
            reward_variance = torch.Tensor([64.0, 64.0])

            for trial in range(num_trials):
                df_trial = df_task[(df_task['trial'] == trial)]
                c = df_trial.choice.item()
                r = df_trial.reward.item()

                # store data
                hybrid_features[trial, 0] = m[0] - m[1]
                hybrid_features[trial, 1] = torch.sqrt(s[0]) - torch.sqrt(s[1])
                hybrid_features[trial, 2] = (m[0] - m[1]) / (torch.sqrt(s[0] + s[1]))
                hybrid_actions[trial] = c

                # update parameters
                k = s[c] / (s[c] + reward_variance[c])
                err = r - m[c]
                m[c] = m[c] + k * err
                s[c] = s[c] - k * s[c]

            inputs_hybrid.append(hybrid_features[4:])
            targets_hybrid.append(hybrid_actions[4:])

        ids.append(participant_counter * torch.ones(num_trials_per_participant))
        participant_counter = participant_counter + 1

inputs_hybrid = torch.cat(inputs_hybrid, dim=0)
targets_hybrid = torch.cat(targets_hybrid, dim=0)
ids = torch.cat(ids, dim=0).long()
torch.save([inputs_hybrid, targets_hybrid, ids], 'data/hybrid_full_data.pth')
