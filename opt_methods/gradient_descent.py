# Custom descent modeled on:
# 	- https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
# 	- https://discuss.pytorch.org/t/simulated-annealing-custom-optimizer/38609
import torch
from torch.optim.optimizer import Optimizer, required

# Defines gradient descent optimizer with options: noise, AGD(Nesterov momentum), or NCE(negative curvature exploitation)
# TODO: Possibly add weight_decay support (L2 reg)
class GradientDescent(Optimizer):
    # correspondence with parameters in the paper is as follows:
    # learning_rate: eta, noise_r: r, noise_L: L, noise_eps: epsilon
    # momentum: 1-theta, NCE_s: s, NCE_gamma: gamma
	def __init__(self, is_ml, neural_net_params=None, learning_rate=0, noise_r=0.1, noise_L=10, noise_eps=0.1, momentum=0, NCE=False, NCE_s=0.1, NCE_gamma=0.1):
		defaults = dict(learning_rate=learning_rate)
		self.learning_rate=learning_rate
		self.is_ml = is_ml
		if is_ml:
			Optimizer.__init__(self, neural_net_params, defaults)
		self.noise_r = noise_r
		self.noise_L = noise_L
		self.noise_eps = noise_eps
		self.momentum = momentum
		self.NCE = NCE
		self.NCE_s = NCE_s
		self.NCE_gamma = NCE_gamma

	def __setstate__(self, state):
	    super(GradientDescent, self).__setstate__(state)

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
	    
	    for group in self.param_groups:
	        for p in group['params']:
	            print("num in param group", len(self.param_groups))
	            print("num of p in group", len(group['params']))
	            print("size of p", p.size())
	            if p.grad is None:
	                continue
	            param_state = self.state[p]
                # noise
	            if self.noise_r > 0:
	                if 'noise_count' not in param_state:
	                    count = param_state['noise_count'] = 0
	                else:
	                    count = param_state['noise_count']
	                print("*Noise part* grad l2 norm: %.3f, count: %d" % (torch.norm(p.grad, p=2).item(), count))
	                if torch.norm(p.grad, p=2).item() <= self.noise_eps and count >= self.noise_L:
	                    param_state['noise_count'] = 0
	                    radius = torch.pow(torch.rand(1), 1.0/p.numel()).mul(self.noise_r)
	                    gauss = torch.randn(p.size())
	                    normed_gauss = gauss.div(torch.norm(gauss, p=2))
	                    noise = normed_gauss.mul(radius)
	                    print("adding noise with l2 norm:", radius)
	                    p.data.add_(noise)
	                else:
	                    param_state['noise_count'] += 1

                # AGD
	            if self.momentum > 0:
	                if 'momentum_buffer' not in param_state:
	                    buf = param_state['momentum_buffer'] = torch.zeros(p.size())
	                else:
	                    buf = param_state['momentum_buffer']
	                if self.NCE:
	                    xt = torch.clone(p.data).detach()
	                    f_xt = loss.item()
	                p.data.add_(self.momentum, buf)
	                loss = closure()
	                if self.NCE:
	                    yt = torch.clone(p.data).detach()
	                    f_yt = loss.item()
	                    g_yt = torch.clone(p.grad.data).detach()
	                    vt = torch.clone(buf.data).detach()
	                p.data.add_(-group['learning_rate'], p.grad.data)
	                buf.mul_(self.momentum).add_(-group['learning_rate'], p.grad.data)
	            else:
	                p.data.add_(-group['learning_rate'], p.grad.data)
	            
                # NCE
	            if self.NCE:
	                norm_vt = torch.norm(vt, p=2)
	                print("*NCE part* norm_vt: %.3f, f_xt: %.3f, f_yt: %.3f, dot: %.3f, norm: %.3f" % (norm_vt.item(), f_xt, f_yt, g_yt.reshape(-1).dot((xt-yt).reshape(-1)).item(), torch.norm((xt-yt), p=2).pow(2).item()))
	                if norm_vt > 0 and f_xt <= f_yt + g_yt.reshape(-1).dot((xt-yt).reshape(-1)) - self.NCE_gamma / 2 * (torch.norm((xt-yt), p=2).pow(2)):
	                    if norm_vt >= self.NCE_s:
	                        p.data = xt
	                        print("setting x_{t+1} = xt")
	                    else:
	                        delta = vt.mul(self.NCE_s).div(norm_vt)
	                        p.data = xt + delta
	                        loss_ = closure()
	                        p.data = xt - delta
	                        loss = closure()
	                        if (loss_ < loss):
	                            print("setting x_{t+1} = xt + delta")
	                            p.data = xt + delta
	                            loss = closure()
	                        else:
	                            print("setting x_{t+1} = xt - delta")
	    return loss

	def step_not_ml(self, f, x):
		assert not self.is_ml
		return x - f.grad(x) * self.learning_rate
