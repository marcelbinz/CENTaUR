import argparse
import torch
import torch.nn.functional as F
from torch.distributions import Binomial

models = ["7B", "13B", "30B", "65B"]
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=models)
args = parser.parse_args()

inputs, num_B_choices = torch.load('data/model=' + args.model + '_full_data.pth')
num_choices = torch.ones_like(num_B_choices)
weights = torch.load('../last_layer_' + args.model + '.pth', map_location='cpu').float()[[29896, 29906]]

probs = F.softmax(inputs @ weights.t(), dim=-1)[:, 1]

llf = -Binomial(total_count=num_choices, probs=probs).log_prob(num_B_choices).sum().item()

torch.save([llf, probs], 'data/loo_llama_fixed_model=' + args.model + '.pth')

print(llf)
