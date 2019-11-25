import torch

from opt_methods.gradient_descent import GradientDescent
from ml_testing.mnist_testing import MnistTest
from ml_testing.mf_testing import MatrixFactorizationTest
from simple_testing.quadratic_function import Quadratic
from simple_testing.cosine_function import Cosine
from simple_testing.simple_testing import run


"""
Top-level script that can be used to test a particular optimizer on a particular function.
Training logs are saved locally.
"""

# ================
# NON-ML FUNCTIONS
# ================
gd_function_optimizer = GradientDescent(None, is_ml=False, lr=0.1)
#run(Quadratic(), gd_function_optimizer, epochs=100, epsilon=0.1, verbosity=True)
run(Cosine(), gd_function_optimizer, epochs=100, epsilon=0.1, verbosity=True)



# ============
# ML FUNCTIONS
# ============
mnist_ffnn_test = MnistTest(ff=True)
mnist_cnn_test = MnistTest(ff=False)
mf_test = MatrixFactorizationTest(load_model=False)

gd_ffnn_optimizer = GradientDescent(
    mnist_ffnn_test.network.parameters(),
    is_ml=True,
    lr=0.1,
    noise_r=0,
    momentum=0,
    NCE=False
)

gd_cnn_optimizer = GradientDescent(
    mnist_cnn_test.network.parameters(),
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
    lr=10,
    noise_r=0,
    momentum=0,
    NCE=False
)

#mnist_ffnn_test.run(1, torch.optim.Adam(mnist_ffnn_test.network.parameters()), sgd=True)
# mnist_cnn_test.run(1, torch.optim.Adam(mnist_cnn_test.network.parameters()), sgd=True)

#mnist_ffnn_test.run(20, gd_ffnn_optimizer, sgd=False)
#mnist_cnn_test.run(1, gd_cnn_optimizer, sgd=False)

#mf_test.run(1, torch.optim.SGD(mf_test.model.parameters(), lr=0.01), sgd=True, save_model=False, log=True)
mf_test.run(1, mf_gd_optim, sgd=False, save_model=False, log=True)
