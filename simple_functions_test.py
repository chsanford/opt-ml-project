from opt_methods.gradient_descent import GradientDescent
from simple_testing.cosine_function import Cosine
from simple_testing.octopus_function import Octopus
from simple_testing.quadratic_function import Quadratic
from simple_testing.simple_testing import run
from simple_testing.simple_testing import run_trials


"""
Top-level script that runs the non-ML experiments in the report.
Plots are output at the end of training.
"""

# vanilla_gd_function_optimizer = GradientDescent(False, learning_rate=0.1, noise_r=0, noise_T=0, noise_eps=0,
#                  momentum=0, NCE=False, NCE_s=0, NCE_gamma=0)
default_agd_function_optimizer = GradientDescent(False)
# run(Quadratic(dim=10), 100, gd_function_optimizer, 0.1, False)
# run(Cosine(dim=10), 100, gd_function_optimizer, 0.1, False)
# run(Octopus(), 100, gd_function_optimizer, 0.1, True)
run_trials(Octopus(), default_agd_function_optimizer)
