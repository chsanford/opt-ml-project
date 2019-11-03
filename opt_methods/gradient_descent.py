# Custom descent modeled on:
# 	- https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
# 	- https://discuss.pytorch.org/t/simulated-annealing-custom-optimizer/38609
from torch.optim.optimizer import Optimizer, required

# Defines vanilla gradient descent optimizer
class GradientDescent(Optimizer):
	def __init__(self, is_ml, neural_net_params=None, learning_rate=0.1):
		defaults = dict(learning_rate=learning_rate)
		# super(GradientDescent, self).__init__(params, defaults)
		self.learning_rate=learning_rate
		self.is_ml = is_ml
		if is_ml:
			Optimizer.__init__(self, neural_net_params, defaults)

	def step(self, closure=None):
		assert self.is_ml
		loss = None
		if closure is not None:
			loss = closure()
		
		for group in self.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue
				p.data.add_(-group['learning_rate'], p.grad.data)

		return loss

	def step_not_ml(self, f, x):
		assert not self.is_ml
		return x - f.grad(x) * self.learning_rate
