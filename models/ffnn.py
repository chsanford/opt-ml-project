import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNN(nn.Module):
    def __init__(self, log_prob=True):
        super(FFNN, self).__init__()
        self.log_prob = log_prob
        self.fc1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1) if self.log_prob else F.softmax(x, dim=1)
