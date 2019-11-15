import torch

from opt_methods.gradient_descent import GradientDescent
from ml_testing.mnist_testing import MnistTest
from ml_testing.mf_testing import MatrixFactorizationTest
from simple_testing.quadratic_function import Quadratic
from simple_testing.simple_testing import run

# A simple test of vanilla gradient descent on f(x) = x_1^2 + x_2^2 for a random initialization
# gd_function_optimizer = GradientDescent(False, learning_rate=0.1)
# run(Quadratic(), 100, gd_function_optimizer, 0.1, True)

mnist_ffnn_test = MnistTest(ff=True)
mnist_cnn_test = MnistTest(ff=False)
mf_test = MatrixFactorizationTest()

gd_ml_optimizer = GradientDescent(
    mnist_ffnn_test.network.parameters(),
    is_ml=True,
    lr=0.1,
    noise_r=0,
    momentum=0,
    NCE=False
)

gd_ml_all_optimizer = GradientDescent(
    mnist_ffnn_test.network.parameters(),
    is_ml=True,
    lr=0.1,
    noise_r=0.1,
    momentum=0.5,
    NCE=True
)

mf_gd_optim = GradientDescent(
    mf_test.model.parameters(),
    is_ml=True,
    lr=0.1,
    noise_r=0,
    momentum=0,
    NCE=False
)

# mnist_ffnn_test.run(1, torch.optim.Adam(mnist_ffnn_test.network.parameters()), sgd=True)
# mnist_cnn_test.run(1, torch.optim.Adam(mnist_cnn_test.network.parameters()), sgd=True)

mnist_ffnn_test.run(20, gd_ml_optimizer, sgd=False)
#mnist_cnn_test.run(20, gd_ml_optimizer, sgd=False)

#mf_test.run(10, torch.optim.SGD(mf_test.model.parameters(), lr=0.01), sgd=True)
#mf_test.run(10, mf_gd_optim, sgd=False)
