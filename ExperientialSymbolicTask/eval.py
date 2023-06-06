import torch
import math

loos = torch.Tensor([torch.load('data/loo_centaur_fixed_model=65B_fold=' + str(i) + '.pth', map_location='cpu')[0] for i in range(8)])
alphas = torch.Tensor([torch.load('data/loo_centaur_fixed_model=65B_fold=' + str(i) + '.pth', map_location='cpu')[1] for i in range(8)])
temps = torch.Tensor([torch.load('data/loo_centaur_fixed_model=65B_fold=' + str(i) + '.pth', map_location='cpu')[2] for i in range(8)])
print(alphas)
print(temps)
print(loos.sum())

ee_inputs, ee_num_B_choices = torch.load('data/model=65B_full_data.pth')
print(ee_inputs.shape)
print(-ee_num_B_choices.shape[0] * math.log(0.5))
