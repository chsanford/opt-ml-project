import numpy as np
import itertools

from opt_methods.gradient_descent import GradientDescent
from simple_testing.simple_testing import is_first_order_stationary_point, is_second_order_stationary_point

from simple_testing.quadratic_function import Quadratic 
from simple_testing.cosine_function import Cosine 
from simple_testing.octopus_function import Octopus 


"""
Top-level script that runs a grid search on the non-ML functions, printing the results to console.
The parameter ranges we tested are at the bottom.
"""

# Generates the parameter ranges.
# Linear is straightforward; if log10, then min and max will be chosen to be the closest exp. of 10
# (e.g. [0.0005, 2] -> [0.001, 0.01, 0.1, 1])
def interpolate(min, max, num_imdt_values):
    assert min < max
    if num_imdt_values is not None:  # linear search
        assert num_imdt_values >= 0
        step = (max - min) / (num_imdt_values + 1)
        return [min + i * step for i in range(num_imdt_values + 2)]
    else:  # log10 search
        min_exp = int(np.ceil(np.log10(min)))
        max_exp = int(np.floor(np.log10(max)))
        return [10 ** i for i in range(min_exp, max_exp + 1)]


# Runs the grid search.
# function: function to be minimized
# fixed_params is a dict of scalars, search_params is a dict ((min, max), num_imdt_values)
# set num_imdt_values >= 0 for linear search, None for log10 search
def grid_search(function, fixed_params=dict(), search_params=dict(),
                max_epochs=200, num_runs=10, metric='loss', eps=0.1, verbose=True):
    searchs = dict()
    fixed_params.update({'noise_eps':eps})
    for p, t in search_params.items():
        searchs[p] = interpolate(t[0][0], t[0][1], t[1])
    keys, values = zip(*searchs.items())
    params = [dict(zip(keys, v)) for v in itertools.product(*values)]
    loss_means = []
    loss_stds = []
    first_means = []
    first_stds = []
    second_means = []
    second_stds = []

    for kwargs in params:
        kwargs.update(fixed_params)
        if verbose:
            print(f'Optimizing with parameters {kwargs}')
        optimizer = GradientDescent(None, is_ml=False, is_verbose=False, **kwargs)
        loss = []
        firsts = []
        seconds = []
        for j in range(num_runs):
            first = second = max_epochs
            if verbose:
                print(f'Run [{j+1}/{num_runs}]')
                print(f'{"epochs":>12}{"loss":>12}')
                print(f'{"--------":>12}{"--------":>12}')
            x = function.random_init()
            print(f'{"0":>12}{function.eval(x):>12.4f}')
            for step in range(max_epochs):
                try:
                    x = optimizer.step_not_ml(function, x)
                    if first == max_epochs and is_first_order_stationary_point(function, x, eps):
                        first = step + 1
                    if second == max_epochs and is_second_order_stationary_point(function, x, eps):
                        second = step + 1
                except:
                    x = function.random_init()
                    first = second = max_epochs
                    break
                if verbose:
                    print(f'{step+1:>12}{function.eval(x):>12.4f}')
            loss.append(function.eval(x)) 
            firsts.append(first)
            seconds.append(second)
        loss_means.append(np.mean(loss))
        loss_stds.append(np.std(loss))
        first_means.append(np.mean(firsts))
        first_stds.append(np.std(firsts))
        second_means.append(np.mean(seconds))
        second_stds.append(np.std(seconds))

    print("Best parameters set found on development set:")
    if metric == 'first':
        best_param = params[np.argmin(first_means)]
    elif metric == 'second':
        best_param = params[np.argmin(second_means)]
    else:
        best_param = params[np.argmin(loss_means)]
    print(best_param)
    print()
    print("Grid scores on development set:")
    for i in range(len(params)):
        print(f'loss: {loss_means[i]:.3f} (+/-{loss_stds[i]:.3f}), first: {first_means[i]:.3f} (+/-{first_stds[i]:.3f}), second: {second_means[i]:.3f} (+/-{second_stds[i]:.3f}) for {params[i]}')

    return best_param


# Wrapper allows for sequential searches over the parameters.
# max_epochs: max number of steps by the optimizer
# num_runs: number of runs to average over
# metric: optimize over which metric, should be 'loss' or 'first' or 'second'
def run(function, searches, fixed_params=dict(), max_epochs=20, num_runs=3, metric='loss', eps=0.1, verbose=True):
    print(f'{function}, running searches {searches} with fixed params {fixed_params}')

    for i, search_params in enumerate(searches):
        s = f'Search [{i+1}/{len(searches)}] {search_params} with fixed params {fixed_params}'
        print(f"{''.ljust(len(s), '=')}\n{s}\n{''.ljust(len(s), '=')}\n")

        params = grid_search(function, fixed_params=fixed_params, search_params=search_params,
                             max_epochs=max_epochs, num_runs=num_runs, metric=metric, eps=eps, verbose=verbose)
        fixed_params.update(params)

    print(f'\nAll done, final parameters: {fixed_params}')


fixed_1 = {'noise_r': 0, 'momentum': 0, 'NCE': False, 'noise_eps': 0.1}
searches_1_1 = [{'lr': [(0.0001, 10), None]}]
searches_1_2 = [{'lr': [(0.04, 0.5), 22]}]
searches_1_3 = [{'lr': [(0.1, 0.14), 3]}]
fixed_2 = {'noise_r': 0, 'NCE': False, 'noise_eps': 0.1}
searches_2_1 = [{'lr': [(0.0001, 10), None], 'momentum': [(0.1,0.9), 7]}]
searches_2_2 = [{'lr': [(0.04, 0.5), 22], 'momentum': [(0.1,0.9), 7]}]
searches_2_3 = [{'lr': [(0.06, 0.14), 7], 'momentum': [(0.5, 0.7), 3]}]
fixed_3 = {'lr': 0.08, 'momentum':0.65, 'NCE': False, 'noise_eps': 0.1}
searches_3_1 = []

#1_1
#run(Octopus(), searches_1_1, fixed_params=fixed_1, max_epochs=500, num_runs=10, metric='second', eps=0.1, verbose=True)
#1_2
#run(Octopus(), searches_1_2, fixed_params=fixed_1, max_epochs=200, num_runs=20, metric='second', eps=0.1, verbose=True)
#1_3
#run(Octopus(), searches_1_3, fixed_params=fixed_1, max_epochs=200, num_runs=50, metric='second', eps=0.1, verbose=True)
### choose lr=0.13 for GD
#2_1
#run(Octopus(), searches_2_1, fixed_params=fixed_2, max_epochs=200, num_runs=10, metric='second', eps=0.1, verbose=True)
#2_2
#run(Octopus(), searches_2_2, fixed_params=fixed_2, max_epochs=200, num_runs=20, metric='second', eps=0.1, verbose=True)
#2_3
#run(Octopus(), searches_2_3, fixed_params=fixed_2, max_epochs=200, num_runs=50, metric='second', eps=0.1, verbose=True)
### choose lr=0.08, momentum=0.65 for AGD
