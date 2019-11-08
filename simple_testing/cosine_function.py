import math
import numpy as np

# f(x) = cos(x1) + cos(x2) + ... + cos(xd)
class Cosine:
	def __init__(self, dim=2):
		self.dim = dim

	def get_name(self):
		return "Cosine-" + str(self.dim)

	# x is a numpy array of length dim
	def eval(self, x):
		assert len(x) == self.dim
		return np.cos(x).sum()

	def grad(self, x):
		assert len(x) == self.dim
		return -1 * np.sin(x)

	def hessian(self, x):
		return np.diag(-1 * np.cos(x))

	def random_init(self):
		return np.random.rand(self.dim) * 4 * math.pi - 2 * math.pi

	def as_string(self):
		return "f(x) = cos(x1) + cos(x2) + ... + cos(xd)"
