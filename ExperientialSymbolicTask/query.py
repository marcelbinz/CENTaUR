import sys

sys.path.append('..')

import time
import argparse
from inference import LLaMAInference
import pandas as pd
import numpy as np
import torch

def replace_right(source, target, replacement, replacements=1):
    return replacement.join(source.rsplit(target, replacements))

if __name__ == "__main__":
    models = ["7B", "13B", "30B", "65B"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=models)
    args = parser.parse_args()

    # loading
    start_loading = time.time()
    llama = LLaMAInference(args.llama_path, args.model, max_batch_size=2)
    print(f"Loaded model {args.model} in {time.time() - start_loading:.2f} seconds")

    # generation
    start_generation = time.time()

    df = pd.read_csv('data/block_complete.csv')
    df['sub'] = pd.factorize(df['sub'])[0]

    question = "Q: Which machine do you choose?\n"\
        "A: Machine"

    subject_counter = 0
    for subject in range(df['sub'].max() + 1):
        df_subject = df[(df['sub'] == subject) & (df['catch'] != 1) & (df['sess'] >= 0)]

        if len(df_subject) == 216:
            df_subject_le = df_subject[df_subject['elic'] == -1]
            df_subject_es = df_subject[df_subject['elic'] == 0]
            df_subject_es['trial'] = pd.factorize(df_subject_es['trial'])[0]

            llama_features = torch.zeros(len(df_subject_es), llama.generator.model.params.dim)
            human_actions = torch.zeros(len(df_subject_es))

            histories = {
                "0.1": [],
                "0.2": [],
                "0.3": [],
                "0.4": [],
                "0.6": [],
                "0.7": [],
                "0.8": [],
                "0.9": []
            }

            for trial in range(int(df_subject_le['trial'].max()) + 1):
                df_trial_le = df_subject_le[df_subject_le['trial'] == trial]
                if df_trial_le['cho'].item() == 1.0:
                    histories[str(df_trial_le['p1'].item())].append(df_trial_le['out'].item())
                    histories[str(df_trial_le['p2'].item())].append(df_trial_le['cfout'].item())
                else:
                    histories[str(df_trial_le['p1'].item())].append(df_trial_le['cfout'].item())
                    histories[str(df_trial_le['p2'].item())].append(df_trial_le['out'].item())

            for trial in range(df_subject_es['trial'].max() + 1):
                df_trial_es = df_subject_es[df_subject_es['trial'] == trial]

                text_B = "Machine 2 delivers -1.0 dollars with " + str(100 - round(df_trial_es.p2.item() * 100, 4)) + "% chance, 1.0 dollars with " + str(round(df_trial_es.p2.item() * 100, 4)) + "% chance.\n\n"
                text_B = replace_right(text_B, ',', ', and')

                hist = histories[str(df_trial_es['p1'].item())]
                text_A = "You made the following observations in the past:\n"
                for item in hist:
                    text_A += "- Machine 1 delivered " + str(item) + " dollars.\n"

                task_description = "Your goal is to maximize the amount of received dollars.\n\n"

                query = text_A + "\n" + text_B + task_description + question
                print('===================')

                results, _ = llama.generate([query], temperature=0.0, top_p=1, max_length=1)
                llama_features[trial] = llama.generator.model.hl.squeeze().detach().cpu()
                human_actions[trial] = df_trial_es['cho'].item() - 1

            torch.save([llama_features, human_actions], 'data/model=' + str(args.model) + '_participant=' + str(subject_counter) + '.pth')
            subject_counter += 1
