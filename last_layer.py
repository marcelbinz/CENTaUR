from inference import LLaMAInference
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--llama-path", type=str, required=True)

args = parser.parse_args()

models = ["7B", "13B", "30B", "65B"]

for model in models:
    llama = LLaMAInference(args.llama_path, model, max_batch_size=2)
    torch.save(llama.generator.model.output.weight.data, 'data/last_layer_' + model + '.pth')
