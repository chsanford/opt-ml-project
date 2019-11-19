import numpy as np
import itertools

from opt_methods.gradient_descent import GradientDescent

from simple_testing.quadratic_function import Quadratic 
from simple_testing.cosine_function import Cosine 
from simple_testing.octopus_function import Octopus



# linear is straightforward; if log10, then min and max will be chosen to be the closest exp. of 10 (e.g. [0.0005, 2] -> [0.001, 0.01, 0.1, 1])
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


# function: function to be minimized
# fixed_params is a dict of scalars, search_params is a dict ((min, max), num_imdt_values)
# set num_imdt_values >= 0 for linear search, None for log10 search
def grid_search(function, fixed_params=dict(), search_params=dict(), max_epochs=200, num_runs=10, verbose=True):
    searchs = dict()
    for p, t in search_params.items():
        searchs[p] = interpolate(t[0][0], t[0][1], t[1])
    keys, values = zip(*searchs.items())
    params = [dict(zip(keys, v)) for v in itertools.product(*values)]
    means = []
    stds = []

    for kwargs in params:
        kwargs.update(fixed_params)
        if verbose:
            print(f'Optimizing with parameters {kwargs}')
        optimizer = GradientDescent(None, is_ml=False, is_verbose=False, **kwargs)
        loss = []
        for j in range(num_runs):
            if verbose:
                print(f'Run [{j+1}/{num_runs}]')
                print(f'{"epochs":>12}{"loss":>12}')
                print(f'{"--------":>12}{"--------":>12}')
            x = function.random_init()
            print(f'{"0":>12}{function.eval(x):>12.4f}')
            for step in range(max_epochs):
                x = optimizer.step_not_ml(function, x)
                if verbose:
                    print(f'{step+1:>12}{function.eval(x):>12.4f}')
            loss.append(function.eval(x)) 
        means.append(np.mean(loss))
        stds.append(np.std(loss))

    print("Best parameters set found on development set:")
    best_param = params[np.argmin(means)]
    print(best_param)
    print()
    print("Grid scores on development set:")
    for mean, std, param in zip(means, stds, params):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, param))

    return best_param


# Wrapper allows for sequential searches over the parameters
# max_epochs: max number of steps by the optimizer
# num_runs: number of runs to average over
def run(function, searches, fixed_params=dict(), max_epochs=20, num_runs=3, verbose=True):
    print(f'{function}, running searches {searches} with fixed params {fixed_params}')

    max_epochs = 10
    num_runs = 3

    for i, search_params in enumerate(searches):
        s = f'Search [{i+1}/{len(searches)}] {search_params} with fixed params {fixed_params}'
        print(f"{''.ljust(len(s), '=')}\n{s}\n{''.ljust(len(s), '=')}\n")

        params = grid_search(function, fixed_params=fixed_params, search_params=search_params,
                             max_epochs=max_epochs, num_runs=num_runs, verbose=verbose)
        fixed_params.update(params)

    print(f'\nAll done, final parameters: {fixed_params}')


seq_searches = [{'lr': [(0.01, 0.1), 0]},
                {'momentum': [(0.01, 0.1), 0]}]

sim_searches = [{'lr': [(0.1, 0.5), 3],
                 'momentum': [(0.1, 0.5), 3]}]

run(Quadratic(), sim_searches, max_epochs=10, num_runs=3, verbose=True)
