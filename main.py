from opt_methods.gradient_descent import GradientDescent
from ml_testing.mnist_testing import MnistTest
from simple_testing.quadratic_function import Quadratic
from simple_testing.octopus_function import Octopus
from simple_testing.simple_testing import run

eps = 0.1

# A simple test of vanilla gradient descent on f(x) = x_1^2 + x_2^2 for a random initialization
gd_function_optimizer = GradientDescent(False, learning_rate=0.1)
gd_function_all_optimizer = GradientDescent(
    False, 
    learning_rate = 0.1,
    noise_r = 0.1,
	noise_eps = eps,
    momentum = 0.3,
    NCE = True,
    NCE_s = 4.9 # (1-0.3)^2/0.1
)
#run(Quadratic(), 100, gd_function_optimizer, 0.1, True)
#run(Quadratic(), 100, gd_function_all_optimizer, 0.1, True)
run(Octopus(), 100, gd_function_optimizer, eps, True)
run(Octopus(), 100, gd_function_all_optimizer, eps, True)

# A simple test of vanilla gradient descent on the MNIST dataset
mnist_test = MnistTest()
gd_ml_optimizer = GradientDescent(True, neural_net_params = mnist_test.network.parameters(), learning_rate = 0.1,)
gd_ml_all_optimizer = GradientDescent(
	True,
	neural_net_params = mnist_test.network.parameters(),
	learning_rate = 0.1,
	noise_r = 0.1,
	momentum = 0.3,
    NCE = True
)

#mnist_test.run(20, gd_ml_optimizer)
mnist_test.run(20, gd_ml_all_optimizer)
