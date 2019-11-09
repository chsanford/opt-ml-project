import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, n, m, r):
        super().__init__()
        self.U = nn.Embedding(n, r)
        self.V = nn.Embedding(m, r)

    def forward(self, idxs):
        u_idxs, v_idxs = idxs[:, 0], idxs[:, 1]
        return (self.U(u_idxs) * self.V(v_idxs)).sum(1)
