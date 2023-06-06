import sys

sys.path.append('..')

import time
import argparse
from inference import LLaMAInference
import numpy as np
import pandas as pd
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

    c13k_problems = pd.read_json("data/c13k_problems.json", orient='index')

    llama_features = torch.zeros(len(c13k_problems), llama.generator.model.params.dim)

    for index in np.arange(len(c13k_problems)):

        value_A = 0
        text_A = "Machine 1 delivers "
        for item_A in c13k_problems.iloc[index].A:
            value_A += item_A[1] * item_A[0]
            text_A += str(item_A[1]) + " dollars with " + str(round(item_A[0] * 100, 4)) + "% chance, "
        text_A = text_A[:-2]
        text_A = replace_right(text_A, ',', ', and')
        text_A += ".\n"

        value_B = 0
        text_B = "Machine 2 delivers "
        for item_B in c13k_problems.iloc[index].B:
            value_B += item_B[1] * item_B[0]
            text_B += str(item_B[1]) + " dollars with " + str(round(item_B[0] * 100, 4)) + "% chance, "
        text_B = text_B[:-2]
        text_B = replace_right(text_B, ',', ', and')
        text_B += ".\n\n"

        text = text_A
        text += text_B

        text += "Your goal is to maximize the amount of received dollars.\n\n"

        text += "Q: Which machine do you choose?\n"

        text += "A: Machine"

        print(text)
        print('=======')

        results, _ = llama.generate([text], temperature=0.0, top_p=1, max_length=1)
        llama_features[index] = llama.generator.model.hl.squeeze().detach().cpu()

    torch.save([llama_features], 'data/model=' + str(args.model) + '.pth')
