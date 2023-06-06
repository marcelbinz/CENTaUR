import sys

sys.path.append('..')

import time
import argparse
from inference import LLaMAInference
import numpy as np
import pandas as pd
import torch

num2words = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six'}

if __name__ == "__main__":
    models = ["7B", "13B", "30B", "65B"]
    datasets = ["exp1", "exp2"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=models)
    parser.add_argument("--dataset", type=str, required=True, choices=datasets)
    args = parser.parse_args()

    # loading
    start_loading = time.time()
    llama = LLaMAInference(args.llama_path, args.model, max_batch_size=2)
    print(f"Loaded model {args.model} in {time.time() - start_loading:.2f} seconds")

    # generation
    start_generation = time.time()

    df = pd.read_csv('data/' + args.dataset + '.csv')

    num_participants = df.participant.max() + 1
    num_tasks = df.task.max() + 1

    instructions = "You made the following observations in the past:\n"

    question = "Q: Which machine do you choose?\n"\
        "A: Machine"

    for participant in range(num_participants):
        df_participant = df[(df['participant'] == participant)]
        for task in range(num_tasks):
            df_task = df_participant[(df_participant['task'] == task)]
            history = ""
            num_trials = df_task.trial.max() + 1

            llama_features = torch.zeros(num_trials, llama.generator.model.params.dim)
            human_actions = torch.zeros(num_trials)

            for trial in range(num_trials):
                df_trial = df_task[(df_task['trial'] == trial)]
                if not df_trial['forced_choice'].item():
                    trials_left = num_trials - trial
                    if trials_left > 1:
                        trials_left = num2words[trials_left] + " additional choices"
                    else:
                        trials_left = num2words[trials_left] + " additional choice"

                    trials_left_string = "Your goal is to maximize the sum of received dollars within " +  trials_left + ".\n\n"

                    prompt = instructions + history + "\n" + trials_left_string + question
                    print(prompt, flush=True)
                    print('=========================')
                    results, _ = llama.generate([prompt], temperature=0.0, top_p=1, max_length=1)
                    llama_features[trial] = llama.generator.model.hl.squeeze().detach().cpu()

                c = df_trial.choice.item()
                r = df_trial.reward.item()

                human_actions[trial] = c
                if c == 0:
                    history += "- Machine 1 delivered " + str(r) + " dollars.\n"
                elif c == 1:
                    history += "- Machine 2 delivered " + str(r) + " dollars.\n"

            torch.save([llama_features, human_actions], 'data/model=' + str(args.model) + '_data=' + str(args.dataset) + '_participant=' + str(participant) + '_task=' + str(task) + '.pth')

    print(df)
