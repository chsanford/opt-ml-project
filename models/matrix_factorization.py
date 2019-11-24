import torch
import torch.nn as nn


"""
Matrix Factorization.
"""
class MatrixFactorization(nn.Module):
    # Must specify dimensions of the score matrix (n x m), and the desired rank of the approx.
    def __init__(self, n, m, r):
        super().__init__()
        self.U = nn.Embedding(n, r)
        self.V = nn.Embedding(m, r)

    # Specify the row and column index.
    def forward(self, idxs):
        u_idxs, v_idxs = idxs[:, 0], idxs[:, 1]
        return (self.U(u_idxs) * self.V(v_idxs)).sum(1)

    def reset_parameters(self):
        self.U.reset_parameters()
        self.V.reset_parameters()
