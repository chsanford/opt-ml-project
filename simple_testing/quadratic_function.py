import numpy as np

# f(x) = x1^2 + x2^2 + ... + xd^2
class Quadratic:
	def __init__(self, dim=2):
		self.dim = dim

	def get_name(self):
		return "Quadratic-" + str(self.dim)

	# x is a numpy array of length dim
	def eval(self, x):
		assert len(x) == self.dim
		return np.square(x).sum()

	def grad(self, x):
		assert len(x) == self.dim
		return 2 * x

	def hessian(self, x):
		return 2 * np.identity(self.dim)

	def random_init(self):
		return np.random.rand(self.dim) * 200 - 100

	def as_string(self):
		return "f(x) = x1^2 + x2^2 + ... + xd^2"
