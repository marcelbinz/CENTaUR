import torch
import matplotlib.pyplot as plt
import seaborn as sns
import math
from torch.distributions import Binomial

_, num_choices, num_B_choices = torch.load('data/model=65B_full_data_choices13k.pth')
num_B_choices = num_B_choices.round()

random = -Binomial(total_count=num_choices, probs=0.5 * torch.ones_like(num_choices)).log_prob(num_B_choices).sum().item()
llama = torch.Tensor([torch.load('data/loo_llama_model=65B_fold=' + str(i) + '.pth', map_location='cpu')[0] for i in range(100)]).sum().item()
centaur = torch.Tensor([torch.load('data/loo_centaur_model=65B_fold=' + str(i) + '.pth', map_location='cpu')[0] for i in range(100)]).sum().item()
beast = torch.load('data/loo_beast.pth')

loos = [random, llama, centaur, beast]
print(loos)

plt.rcParams["figure.figsize"] = (1.9,2.2)
plt.bar(['Random', 'LLaMA', 'CENTaUR', 'BEAST'], loos, color=['C0', 'C0', 'C1', 'C0'], alpha=0.8)
plt.ylim(0.95*min(loos), 1.02*max(loos))
sns.despine()
plt.ylabel('Negative log-likelihood', size=9)
plt.ylim(30000, 120000)
plt.yticks([30000, 60000, 90000, 120000])
plt.xticks(rotation='vertical', size=9)
plt.tight_layout()
plt.yticks(size=9)
plt.savefig('figures/choices13k_loos.pdf', bbox_inches='tight')
plt.show()
