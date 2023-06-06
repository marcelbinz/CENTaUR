import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from tqdm import tqdm
from torch.distributions import Binomial

class BinomialRegression(nn.Module):
    def __init__(self, num_inputs, alpha=0):
        super().__init__()
        self.num_inputs = num_inputs
        self.alpha = alpha
        self.W = nn.Linear(num_inputs, 1, bias=False)

    def forward(self, X):
        return self.W(X).squeeze(-1)

    def fit(self, X, num_choices, num_B_choices, num_iterations=100):
        optimizer = optim.LBFGS(self.parameters())

        for i in tqdm(range(num_iterations)):
            def closure():
                optimizer.zero_grad()
                logits = self(X)
                loss = -Binomial(total_count=num_choices, logits=logits).log_prob(num_B_choices).mean() + self.alpha * self.W.weight.pow(2).sum()
                loss.backward()
                return loss
            optimizer.step(closure)

class JointBinomialRegression(nn.Module):
    def __init__(self, num_inputs, alpha, temp):
        super().__init__()
        self.num_inputs = num_inputs
        self.W = nn.Linear(num_inputs, 1, bias=False)

        self.alpha = alpha
        self.temp = temp

    def forward(self, X):
        return ((1.0/self.temp) * self.W(X).squeeze(-1)).clip(-6.0, 6.0)

    def fit(self, dfd_X, dfd_num_choices, dfd_num_B_choices, ht_X, ht_num_choices, ht_num_B_choices, num_iterations=100):
        optimizer = optim.LBFGS(self.parameters())

        for i in tqdm(range(num_iterations)):
            def closure():
                optimizer.zero_grad()
                dfd_logits = self(dfd_X)
                ht_logits = self(ht_X)
                loss_dfd = -Binomial(total_count=dfd_num_choices, logits=dfd_logits).log_prob(dfd_num_B_choices).mean()
                loss_ht = -Binomial(total_count=ht_num_choices, logits=ht_logits).log_prob(ht_num_B_choices).mean()
                loss = 0.5 * loss_dfd + 0.5 * loss_ht + self.alpha * self.W.weight.pow(2).sum()
                loss.backward()
                return loss
            optimizer.step(closure)

class MixedEffectBinomialRegression(nn.Module):
    def __init__(self, num_inputs, alpha=0, num_groups=0):
        super().__init__()
        self.num_inputs = num_inputs
        self.alpha = alpha
        self.W = nn.Linear(num_inputs, 1, bias=False)
        self.W_random = nn.Parameter(0.01 * torch.randn(num_groups, num_inputs))

    def forward(self, X, ids):
        W_random = self.W_random[ids]
        return self.W(X).squeeze(-1) + (X * W_random).sum(-1)

    def fit(self, X, num_choices, num_B_choices, ids, num_iterations=100):
        optimizer = optim.LBFGS(self.parameters())

        for i in tqdm(range(num_iterations)):
            def closure():
                optimizer.zero_grad()
                logits = self(X, ids)
                loss = -Binomial(total_count=num_choices, logits=logits).log_prob(num_B_choices).mean() + self.alpha * (self.W.weight.pow(2).sum() + self.W_random.pow(2).sum())
                loss.backward()
                return loss
            optimizer.step(closure)

class TemperatureBinomialRegression(nn.Module):
    def __init__(self, num_inputs, model):
        super().__init__()
        self.num_inputs = num_inputs
        self.W = torch.load('../last_layer_' + model + '.pth', map_location='cuda').float()[[29896, 29906]] # indicies for tokens 1 and 2
        self.log_temps = nn.Parameter(torch.ones([]))

    def forward(self, X):
        temps = self.log_temps.exp()
        return F.softmax(temps * (X @ self.W.t()), dim=-1)[:, 1]

    def fit(self, X, num_choices, num_B_choices, num_iterations=100):
        optimizer = optim.LBFGS(self.parameters())

        for i in tqdm(range(num_iterations)):
            def closure():
                optimizer.zero_grad()
                probs = self(X)
                loss = -Binomial(total_count=num_choices, probs=probs).log_prob(num_B_choices).mean()
                loss.backward()
                return loss
            optimizer.step(closure)
