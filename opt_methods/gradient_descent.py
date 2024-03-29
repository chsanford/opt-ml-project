import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np
from collections import defaultdict


"""
Defines gradient descent optimizer with options: noise, AGD (Nesterov momentum), or NCE (negative curvature exploitation).
Custom descent modeled on:
- https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
- https://discuss.pytorch.org/t/simulated-annealing-custom-optimizer/38609
"""

class GradientDescent(Optimizer):
    """
    Correspondence with parameters in the paper is as follows:
    # lr: eta, noise_r: r, noise_T: L, noise_eps: epsilon
    # momentum: 1-theta, NCE_s: s, NCE_gamma: gamma (gamma should be theta^2/eta)
    """
    def __init__(self, model_params, is_ml=True, lr=0.1, noise_r=0, noise_T=-1, noise_eps=0,
                 momentum=0, NCE=False, NCE_s=0, NCE_gamma=0, is_verbose=False):
        self.is_ml = is_ml
        if is_ml:
            Optimizer.__init__(self, model_params, dict())
        else:
            self.state = defaultdict(dict)
        self.lr = lr
        self.noise_r = noise_r
        self.noise_T = noise_T
        self.noise_eps = noise_eps
        self.momentum = momentum
        self.NCE = NCE
        self.NCE_s = NCE_s
        self.NCE_gamma = pow((1-momentum), 2) / lr
        self.is_verbose = is_verbose

    def initialize(self, f=None, x=None):
        if self.is_ml:
            params = []
            for group in self.param_groups:
                for p in group['params']:
                    params.append(p)
            param_state = self.state[params[0]]
            param_state['noise_count'] = 0
            for p in params:
                param_state = self.state[p]
                param_state['momentum_buffer'] = torch.zeros(p.size())
        else:
            state = self.state[f.get_name()]
            state['noise_count'] = 0
            state['momentum_buffer'] = np.zeros(x.size)



    # Returns a dictionary of optimizer parameter values.
    def get_params(self):
        params = ['lr', 'momentum', 'noise_T', 'noise_eps', 'noise_r', 'NCE', 'NCE_s', 'NCE_gamma']
        d = dict()
        for p in params:
            d[p] = self.__getattribute__(p)
        return d


    # Corresponds to the step in a torch.optim optimizer, updating the model parameters.
    def step(self, closure=None):
        assert self.is_ml
        # NCE is only defined for AGD
        if self.NCE:
            assert self.momentum > 0
        # need to define closure to use noise, AGD, or NCE
        if self.noise_r > 0 or self.momentum > 0 or self.NCE:
            assert closure != None
        loss = None
        if closure is not None:
            loss = closure()

        params = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                params.append(p)

        # noise
        if self.noise_r > 0:
            param_state = self.state[params[0]]
            if 'noise_count' not in param_state:
                param_state['noise_count'] = 0
            all_grad = torch.cat([p.grad.reshape(-1) for p in params], dim=0)
            all_grad_norm = torch.norm(all_grad, p=2).item()
            if self.is_verbose:
                print("*Noise part* grad l2 norm: %.3f, count: %d" % (all_grad_norm, param_state['noise_count']))
            if all_grad_norm <= self.noise_eps and param_state['noise_count'] >= self.noise_T:
                param_state['noise_count'] = 0
                radius = torch.pow(torch.rand(1), 1.0 / all_grad.numel()).mul(self.noise_r)
                gauss = torch.randn(all_grad.size())
                normed_gauss = gauss.div(torch.norm(gauss, p=2))
                noise = normed_gauss.mul(radius)
                if self.is_verbose:
                    print("add noise with l2 norm:", radius)
                i = 0
                for p in params:
                    p.data.add_(noise[i:i + p.numel()].reshape(p.size()))
                    i = i + p.numel()
            else:
                param_state['noise_count'] += 1
                if self.is_verbose:
                    print("no noise added")

        # AGD
        if self.momentum > 0:
            if self.NCE:
                xt = torch.tensor([])
                yt = torch.tensor([])
                g_yt = torch.tensor([])
                vt = torch.tensor([])
            for p in params:
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros(p.size())
                else:
                    buf = param_state['momentum_buffer']
                if self.NCE:
                    xt = torch.cat([xt, torch.clone(p.data).detach().reshape(-1)], dim=0)
                    vt = torch.cat([vt, torch.clone(buf.data).detach().reshape(-1)], dim=0)
                    f_xt = loss.item()
                p.data.add_(self.momentum, buf)
            loss = closure()
            for p in params:
                buf = self.state[p]['momentum_buffer']
                if self.NCE:
                    yt = torch.cat([yt, torch.clone(p.data).detach().reshape(-1)], dim=0)
                    g_yt = torch.cat([g_yt, torch.clone(p.grad.data).detach().reshape(-1)], dim=0)
                    f_yt = loss.item()
                p.data.add_(-self.lr, p.grad.data)
                buf.mul_(self.momentum).add_(-self.lr, p.grad.data)
        else:
            for p in params:
                p.data.add_(-self.lr, p.grad.data)

        # NCE
        def copy_to_p(params, update):
            i = 0
            for p in params:
                p.data = update[i:i + p.numel()].reshape(p.size())
                i = i + p.numel()

        if self.NCE:
            norm_vt = torch.norm(vt, p=2)
            if self.is_verbose:
                print(
                    "*NCE part* f(xt): %.3f, f(yt): %.3f, <grad_f(yt),xt-yt>: %.3f, ||xt-yt||^2: %.3f, ||vt||: %.3f" % (
                    f_xt, f_yt, g_yt.dot((xt - yt)).item(), torch.norm((xt - yt), p=2).pow(2).item(), norm_vt.item()))
            if norm_vt > 0 and f_xt <= f_yt + g_yt.dot((xt - yt)) - self.NCE_gamma / 2 * (
            torch.norm((xt - yt), p=2).pow(2)):
                for p in params:
                    self.state[p]['momentum_buffer'] = torch.zeros(p.size())
                if norm_vt >= self.NCE_s:
                    copy_to_p(params, xt)
                    if self.is_verbose:
                        print("setting x_{t+1} = xt")
                else:
                    delta = vt.mul(self.NCE_s).div(norm_vt)
                    copy_to_p(params, xt + delta)
                    loss_ = closure()
                    copy_to_p(params, xt - delta)
                    loss = closure()
                    if (loss_ < loss):
                        if self.is_verbose:
                            print("setting x_{t+1} = xt + delta")
                        copy_to_p(params, xt + delta)
                        loss = closure()
                    else:
                        if self.is_verbose:
                            print("setting x_{t+1} = xt - delta")
            else:
                if self.is_verbose:
                    print("no change by NCE")
        return loss


    # Given a function f from simple_testing, directly returns the new x from taking a step.
    def step_not_ml(self, f, x):
        assert not self.is_ml
        # NCE is only defined for AGD
        if self.NCE:
            assert self.momentum > 0

        state = self.state[f.get_name()]

        # noise
        if self.noise_r > 0:
            if 'noise_count' not in state:
                state['noise_count'] = 0
            if self.is_verbose:
                print("*Noise part* grad l2 norm: %.3f, count: %d" % (
                np.linalg.norm(f.grad(x), ord=2), state['noise_count']))
            if np.linalg.norm(f.grad(x), ord=2) <= self.noise_eps and state['noise_count'] >= self.noise_T:
                state['noise_count'] = 0
                radius = pow(np.random.uniform(0, 1, 1)[0], 1.0 / x.size) * self.noise_r
                gauss = np.random.normal(0, 1, x.size)
                noise = gauss / np.linalg.norm(gauss, ord=2) * radius
                if self.is_verbose:
                    print("add noise with l2 norm:", radius)
                x = x + noise
            else:
                state['noise_count'] += 1
                if self.is_verbose:
                    print("no noise added")

        # AGD
        if self.momentum > 0:
            if 'momentum_buffer' not in state:
                state['momentum_buffer'] = np.zeros(x.size)
            vt = state['momentum_buffer']
            xt = x
            yt = x + self.momentum * vt
            x = yt - self.lr * f.grad(yt)
            state['momentum_buffer'] = self.momentum * vt - self.lr * f.grad(yt)
        else:
            x = x - self.lr * f.grad(x)

        # NCE
        if self.NCE:
            norm_vt = np.linalg.norm(vt, ord=2)
            if self.is_verbose:
                print(
                    "*NCE part* f(xt): %.3f, f(yt): %.3f, <grad_f(yt),xt-yt>: %.3f, ||xt-yt||^2: %.3f, ||vt||: %.3f" % (
                    f.eval(xt), f.eval(yt), np.dot(f.grad(yt), xt - yt), pow(np.linalg.norm((xt - yt), ord=2), 2),
                    norm_vt))
            if norm_vt > 0 and f.eval(xt) <= f.eval(yt) + np.dot(f.grad(yt), xt - yt) - self.NCE_gamma / 2 * pow(
                    np.linalg.norm((xt - yt), ord=2), 2):
                state['momentum_buffer'] = np.zeros(x.size)
                if norm_vt >= self.NCE_s:
                    x = xt
                    if self.is_verbose:
                        print("setting x_{t+1} = xt")
                else:
                    delta = vt * self.NCE_s / norm_vt
                    if f.eval(xt + delta) < f.eval(xt - delta):
                        x = xt + delta
                        if self.is_verbose:
                            print("setting x_{t+1} = xt + delta")
                    else:
                        x = xt - delta
                        if self.is_verbose:
                            print("setting x_{t+1} = xt - delta")
            else:
                if self.is_verbose:
                    print("no change by NCE")
        return x
