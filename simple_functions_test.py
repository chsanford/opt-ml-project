from opt_methods.gradient_descent import GradientDescent
from simple_testing.cosine_function import Cosine
from simple_testing.octopus_function import Octopus
from simple_testing.quadratic_function import Quadratic
from simple_testing.simple_testing import run


gd_function_optimizer = GradientDescent(False, learning_rate=0.1)
run(Quadratic(dim=10), 100, gd_function_optimizer, 0.1, False)
run(Cosine(dim=10), 100, gd_function_optimizer, 0.1, False)
run(Octopus(), 100, gd_function_optimizer, 0.1, True)



