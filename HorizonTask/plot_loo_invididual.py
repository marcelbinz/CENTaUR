import torch
from torch.distributions import Bernoulli
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from groupBMC.groupBMC import GroupBMC

# LLaMA
unsorted_centaur_probs = torch.sigmoid(torch.stack([torch.load('data/loo_centaur_fixed_model=65B_fold=' + str(i) + '.pth', map_location='cpu')[2] for i in range(100)], dim=-1))
unsorted_hybrid_probs = torch.sigmoid(torch.stack([torch.load('data/loo_hybrid_fixed_fold=' + str(i) + '.pth', map_location='cpu')[2] for i in range(100)], dim=-1))
unsorted_llama_probs = torch.stack([torch.load('data/loo_llama_fixed_model=65B_fold=' + str(i) + '.pth', map_location='cpu')[1] for i in range(100)], dim=-1)
data_points = torch.load('data/splits.pth')

centaur_probs = torch.zeros(data_points.numel())
hybrid_probs = torch.zeros(data_points.numel())
llama_probs = torch.zeros(data_points.numel())
for i in range(data_points.shape[0]):
    for j in range(data_points.shape[1]):
        centaur_probs[data_points[i, j]] = unsorted_centaur_probs[i, j]
        hybrid_probs[data_points[i, j]] = unsorted_hybrid_probs[i, j]
        llama_probs[data_points[i, j]] = unsorted_llama_probs[i, j]

_, num_B_choices, ids = torch.load('data/model=65B_full_data.pth')
loo_centaur = np.zeros(60)
loo_hybrid = np.zeros(60)
loo_random = np.zeros(60)
loo_llama = np.zeros(60)
for i in range(ids.shape[0]):
    loo_centaur[ids[i]] += -Bernoulli(centaur_probs[i]).log_prob(num_B_choices[i]).item()
    loo_hybrid[ids[i]] += -Bernoulli(hybrid_probs[i]).log_prob(num_B_choices[i]).item()
    loo_random[ids[i]] += -Bernoulli(0.5*torch.ones([])).log_prob(num_B_choices[i]).item()
    loo_llama[ids[i]] += -Bernoulli(llama_probs[i]).log_prob(num_B_choices[i]).item()

print(loo_centaur)
print(loo_hybrid)
print(loo_random)
print(loo_llama)

part = 0
if part == 0:
    start = 0
    end = 30
else:
    start = 30
    end = 60

loos = np.array([loo_random, loo_llama, loo_centaur, loo_hybrid])
print('N best described by CENTaUR:')
print((loos.argmin(0) == 2).sum())
result = GroupBMC(-loos).get_result()
print(result.protected_exceedance_probability)
loos = loos - loos.min(0, keepdims=True)
print(loos.shape)

plt.rcParams["figure.figsize"] = (5.5,1.35)

fig, ax = plt.subplots()
divider = make_axes_locatable(ax)
cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
fig.add_axes(cbar_ax)

cmap = mpl.colors.LinearSegmentedColormap.from_list('testCmap', ['#363737', '#ffffff'])
ax = sns.heatmap(loos[:, start:end], cmap=cmap, vmin=0, vmax=10, square=True, linewidths=.5, ax=ax, cbar_ax=cbar_ax, linecolor='#d8dcd6')
fig.axes[-1].tick_params(labelsize=8)
ax.set_yticklabels(['Random', 'LLaMA', 'CENTaUR', 'Hybrid'], rotation='horizontal', size=8)
if part != 0:
    ax.set_xlabel('Participant', size=9)
labels = [int(item.get_text()) + start for item in ax.get_xticklabels()]
print(labels)
ax.set_xticklabels(labels, fontsize=8)


[l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 5 != 0]
plt.tight_layout()
plt.savefig('figures/individual_loos_' + str(part) + '.pdf', bbox_inches='tight')
plt.show()
