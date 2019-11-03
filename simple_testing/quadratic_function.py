import numpy as np

# f(x) = x1^2 + x2^2
class Quadratic():
	# x is a 2-dimensional numpy array
	def eval(self, x):
		return np.square(x).sum()

	def grad(self, x):
		return 2 * x

	def hessian(self, x):
		return 2 * np.identity(2)

	def random_init(self):
		return np.random.rand(2) * 200 - 100

	def as_string(self):
		return "f(x) = x1^2 + x2^2"
