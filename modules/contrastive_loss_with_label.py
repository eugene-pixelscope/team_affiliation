import torch
import torch.nn as nn
import math
from itertools import product


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size, gc_i, gc_j):
        N = 2 * batch_size
        mask = torch.zeros((N, N))
        # ii
        ii_m = torch.zeros((N // 2, N // 2))
        for i, j in product(range(N // 2), repeat=2):
            ii_m[i, j] = gc_i[i] == gc_i[j]
        mask[:N // 2, :N // 2] = ii_m
        # jj
        jj_m = torch.zeros((N // 2, N // 2))
        for i, j in product(range(N // 2), repeat=2):
            jj_m[i, j] = gc_j[i] == gc_j[j]
        mask[N // 2:, N // 2:] = jj_m
        # ij
        ij_m = torch.zeros((N // 2, N // 2))
        for i, j in product(range(N // 2), repeat=2):
            ij_m[i, j] = gc_i[i] == gc_j[j]
        mask = mask.bool()
        mask[:N // 2, N // 2:] = ij_m
        mask[N // 2:, :N // 2] = ij_m.T

        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j, gc_i, gc_j):
        device = z_i.device
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        mask = self.mask_correlated_samples(self.batch_size, gc_i, gc_j).to(device).long()

        positive_samples = (sim * mask).reshape(N, -1)
        negative_samples = (sim * ~mask).reshape(N, -1)

        labels = torch.zeros(N).to(device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss
