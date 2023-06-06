import torch
import matplotlib.pyplot as plt
import seaborn as sns
import math

inputs, num_B_choices, _ = torch.load('data/hybrid_full_data.pth')

centaur = torch.Tensor([torch.load('data/loo_centaur_fixed_model=65B_fold=' + str(i) + '.pth', map_location='cpu')[0] for i in range(100)]).sum()
hybrid = torch.Tensor([torch.load('data/loo_hybrid_fixed_fold=' + str(i) + '.pth', map_location='cpu')[0] for i in range(100)]).sum()
llama = torch.Tensor([torch.load('data/loo_llama_fixed_model=65B_fold=' + str(i) + '.pth', map_location='cpu')[0] for i in range(100)]).sum()
random = -num_B_choices.shape[0] * math.log(1/2)

loos = [random, llama, centaur, hybrid]
print(loos)

plt.rcParams["figure.figsize"] = (1.9,2.2)

plt.bar(['Random', 'LLaMA', 'CENTaUR', 'Hybrid'], loos, color=['C0', 'C0', 'C1', 'C0'], alpha=0.8)
plt.ylim(0.95*min(loos), 1.02*max(loos))
sns.despine()
plt.ylabel('Negative log-likelihood', size=9)
plt.ylim(20000, 50000)
plt.xticks(rotation='vertical', size=9)
plt.tight_layout()
plt.yticks([20000, 30000, 40000, 50000], size=9)
plt.savefig('figures/horizon_loos.pdf', bbox_inches='tight')
plt.show()
