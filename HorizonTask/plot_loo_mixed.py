import torch
import matplotlib.pyplot as plt
import seaborn as sns

centaur_full = torch.Tensor([torch.load('data/loo_centaur_mixed_model=65B_fold=' + str(i) + '.pth', map_location='cpu')[0] for i in range(100)]).sum()
centaur_none = torch.Tensor([torch.load('data/loo_centaur_fixed_model=65B_fold=' + str(i) + '.pth', map_location='cpu')[0] for i in range(100)]).sum()
hybrid_full = torch.Tensor([torch.load('data/loo_hybrid_mixed_fold=' + str(i) + '.pth', map_location='cpu')[0] for i in range(100)]).sum() # TODO
print(centaur_none)
print(centaur_full)
print(hybrid_full)

loos = [centaur_none, centaur_full, hybrid_full]
print(loos)

plt.rcParams["figure.figsize"] = (2.1,2.2)

plt.bar(['CENTaUR', 'CENTaUR (mixed)', 'Hybrid (mixed)'], loos, color=['C0', 'C1', 'C0'], alpha=0.7)
plt.ylim(0.95*min(loos), 1.02*max(loos))
sns.despine()
plt.ylabel('Negative \nlog-likelihood', size=9)
plt.xticks(rotation='vertical', size=9)
plt.tight_layout()
plt.yticks([23000, 24000, 25000, 26000], size=9)
plt.ylim(23000, 26000)
plt.savefig('figures/differences.pdf', bbox_inches='tight')
plt.show()
