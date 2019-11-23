import numpy as np
import math


"""
2-D octopus function from https://arxiv.org/pdf/1705.10412.pdf for restricted domain
"""
class Octopus:
    def __init__(self):
        self.gamma = 1
        self.L = math.e
        self.tau = math.e
        self.S = 2 * self.gamma - 4 * self.L
        self.v = -1 * self.g1(2 * self.tau) + 4 * self.L * self.tau

    def get_name(self):
        return "Octopus-2"

    # x is a numpy array of length 2
    def eval(self, x):
        return self.h(abs(x[0]), abs(x[1]))

    def grad(self, x):
        return np.sign(x) * self.h_grad(abs(x[0]), abs(x[1]))

    def hessian(self, x):
        return np.outer(np.sign(x), np.sign(x)) * self.h_hessian(abs(x[0]), abs(x[1]))

    def random_init(self):
        return np.random.rand(2) * 2 - 1

    def as_string(self):
        return "2-D octopus function from https://arxiv.org/pdf/1705.10412.pdf"

    def h(self, u1, u2):
        gamma = self.gamma
        L = self.L
        tau = self.tau
        v = self.v
        if u1 >= 0 and u1 <= tau and u2 >= 0 and u2 <= tau:
            return -1 * gamma * u1 ** 2 + L * u2 ** 2
        elif u1 >= tau and u1 <= 2 * tau and u2 >= 0 and u2 <= tau:
            return self.g(u1, u2)
        elif u1 >= 2 * tau and u1 <= 6 * tau and u2 >= 0 and u2 <= tau:
            return L * (u1 - 4 * tau) ** 2 - gamma * u2 ** 2 - v
        elif u1 >= 2 * tau and u1 <= 6 * tau and u2 >= tau and u2 <= 2 * tau:
            return L * (u1 - 4 * tau) ** 2 + self.g1(u2) - v
        elif u1 >= 2 * tau and u1 <= 6 * tau and u2 >= 2 * tau and u2 <= 6 * tau:
            return L * (u1 - 4 * tau) ** 2 + L * (u2 - 4 * tau) ** 2 - 2 * v
        else:
            print("ERROR: Out of domain!")
            raise ValueError

    def h_grad(self, u1, u2):
        gamma = self.gamma
        L = self.L
        tau = self.tau
        if u1 >= 0 and u1 <= tau and u2 >= 0 and u2 <= tau:
            return np.asarray([-2 * gamma * u1, 2 * L * u2])
        elif u1 >= tau and u1 <= 2 * tau and u2 >= 0 and u2 <= tau:
            return self.g_grad(u1, u2)
        elif u1 >= 2 * tau and u1 <= 6 * tau and u2 >= 0 and u2 <= tau:
            return np.asarray([2 * L * (u1 - 4 * tau), -2 * gamma * u2])
        elif u1 >= 2 * tau and u1 <= 6 * tau and u2 >= tau and u2 <= 2 * tau:
            return np.asarray([2 * L * (u1 - 4 * tau), self.g1_deriv(u2)])
        elif u1 >= 2 * tau and u1 <= 6 * tau and u2 >= 2 * tau and u2 <= 6 * tau:
            return np.asarray([2 * L * (u1 - 4 * tau), 2 * L * (u2 - 4 * tau)])
        else:
            print("ERROR: Out of domain!")

    def h_hessian(self, u1, u2):
        gamma = self.gamma
        L = self.L
        tau = self.tau
        if u1 >= 0 and u1 <= tau and u2 >= 0 and u2 <= tau:
            return np.diag(np.asarray([-2 * gamma, 2 * L]))
        elif u1 >= tau and u1 <= 2 * tau and u2 >= 0 and u2 <= tau:
            return self.g_hessian(u1, u2)
        elif u1 >= 2 * tau and u1 <= 6 * tau and u2 >= 0 and u2 <= tau:
            return np.diag(np.asarray([2 * L, -2 * gamma]))
        elif u1 >= 2 * tau and u1 <= 6 * tau and u2 >= tau and u2 <= 2 * tau:
            return np.diag(np.asarray([2 * L, self.g1_deriv2(u2)]))
        elif u1 >= 2 * tau and u1 <= 6 * tau and u2 >= 2 * tau and u2 <= 6 * tau:
            return np.diag(np.asarray([2 * L, 2 * L]))
        else:
            print("ERROR: Out of domain!")

    def g(self, u1, u2):
        return self.g1(u1) + self.g2(u1) * u2 ** 2

    def g_grad(self, u1, u2):
        return np.asarray([self.g1_deriv(u1) + self.g2_deriv(u1) * u2 ** 2, 2 * self.g2(u1) * u2])

    def g_hessian(self, u1, u2):
        return np.asarray([
            [self.g1_deriv2(u1) + self.g2_deriv2(u1) * u2 ** 2, 2 * self.g2_deriv(u1) * u2],
            [2 * self.g2_deriv(u1) * u2, 2 * self.g2(u1)]
        ])

    def g1(self, u):
        gamma = self.gamma
        tau = self.tau
        return self.q(u) + self.q(tau) - gamma * tau ** 2

    def g1_deriv(self, u):
        return self.q_deriv(u)

    def g1_deriv2(self, u):
        return self.q_deriv2(u)

    def q(self, u):
        gamma = self.gamma
        L = self.L
        tau = self.tau
        S = self.S
        return -2 * self.gamma * self.tau * u \
               - gamma * (u - tau) ** 2 \
               + (3 * S + 2 * L + 4 * gamma) / (3 * tau) * (u - tau) ** 3 \
               + (2 * S + 2 * L + 2 * gamma) / (4 * tau ** 2) * (u - tau) ** 4

    def q_deriv(self, u):
        gamma = self.gamma
        L = self.L
        tau = self.tau
        S = self.S
        return -2 * gamma * tau \
               - 2 * gamma * (u - tau) \
               + (3 * S + 2 * L + 4 * gamma) / (tau) * (u - tau) ** 2 \
               + (2 * S + 2 * L + 2 * gamma) / (tau ** 2) * (u - tau) ** 3

    def q_deriv2(self, u):
        gamma = self.gamma
        L = self.L
        tau = self.tau
        S = self.S
        return - 2 * gamma \
               + 2 * (3 * S + 2 * L + 4 * gamma) / (tau) * (u - tau) \
               + 3 * (2 * S + 2 * L + 2 * gamma) / (tau ** 2) * (u - tau) ** 2

    def g2(self, u):
        gamma = self.gamma
        L = self.L
        tau = self.tau
        v = self.v
        return -1 * gamma \
               + 10 * (L + gamma) * (u - 2 * tau) ** 3 / (tau ** 3) \
               + 15 * (L + gamma) * (u - 2 * tau) ** 4 / (tau ** 4) \
               + 6 * (L + gamma) * (u - 2 * tau) ** 5 / (tau ** 5)

    def g2_deriv(self, u):
        gamma = self.gamma
        L = self.L
        tau = self.tau
        v = self.v
        return 30 * (L + gamma) * (u - 2 * tau) ** 3 / (tau ** 2) \
               + 60 * (L + gamma) * (u - 2 * tau) ** 4 / (tau ** 3) \
               + 30 * (L + gamma) * (u - 2 * tau) ** 5 / (tau ** 4)

    def g2_deriv2(self, u):
        gamma = self.gamma
        L = self.L
        tau = self.tau
        v = self.v
        return 60 * (L + gamma) * (u - 2 * tau) ** 3 / tau \
               + 180 * (L + gamma) * (u - 2 * tau) ** 4 / (tau ** 2) \
               + 120 * (L + gamma) * (u - 2 * tau) ** 5 / (tau ** 3)
