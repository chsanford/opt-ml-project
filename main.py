from opt_methods.gradient_descent import GradientDescent
from ml_testing.mnist_testing import MnistTest
from simple_testing.quadratic_function import Quadratic
from simple_testing.simple_testing import run

# A simple test of vanilla gradient descent on f(x) = x_1^2 + x_2^2 for a random initialization
gd_function_optimizer = GradientDescent(False, learning_rate=0.1)
run(Quadratic(), 100, gd_function_optimizer, 0.1, True)

# A simple test of vanilla gradient descent on the MNIST dataset
mnist_test = MnistTest()
gd_ml_optimizer = GradientDescent(
	True,
	neural_net_params=mnist_test.network.parameters(),
	learning_rate=0.1
)
mnist_test.run(10, gd_ml_optimizer)