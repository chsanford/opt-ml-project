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
                    print(f'{step+1:>12}{function.eval(x):>12.4f}')
                    if first == max_epochs and is_first_order_stationary_point(function, x, eps):
                        first = step + 1
                    if second == max_epochs and is_second_order_stationary_point(function, x, eps):
                        second = step + 1
                except:
                    x = function.random_init()
                    first = second = max_epochs
            #if verbose:
            #    print(f'{function.eval(x):>12.4f}')
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


seq_searches = [{'lr': [(0.01, 0.1), 0]},
                {'momentum': [(0.01, 0.1), 0]}]

sim_searches = [{'lr': [(0.1, 0.5), 3],
                 'momentum': [(0.1, 0.5), 3]}]

# Octopus function
oct_fixed_gd = {'noise_r': 0, 'momentum': 0, 'NCE': False, 'noise_eps': 0.1}
oct_gd_1 = [{'lr': [(0.0001, 10), None]}]
oct_gd_2 = [{'lr': [(0.04, 0.5), 22]}]
oct_gd_3 = [{'lr': [(0.1, 0.14), 3]}]
### choose lr=0.13 for GD
oct_fixed_gdnoise = {'lr': 0.13, 'momentum':0, 'NCE': False, 'noise_eps': 0.1}
oct_gdnoise_1 = [{'noise_r': [(0.001, 10), None], 'noise_T': [(0, 100), 4]}] #500 runs, r=1, T=0
oct_gdnoise_2 = [{'noise_r': [(0.5, 3), 4], 'noise_T': [(0, 100), 9]}] #500 runs, r=3, T=0
oct_gdnoise_3 = [{'noise_r': [(1, 6), 4], 'noise_T': [(0, 100), 4]}] #500 runs, r=5, T=0
### choose r=5, T=0
### first: 16.394 (+/-9.718), second: 26.016 (+/-9.331)
oct_fixed_agd = {'noise_r': 0, 'NCE': False, 'noise_eps': 0.1}
oct_agd_1 = [{'lr': [(0.0001, 10), None], 'momentum': [(0.1,0.9), 7]}]
oct_agd_2 = [{'lr': [(0.02, 0.3), 13], 'momentum': [(0.1, 0.9), 7]}]
oct_agd_3 = [{'lr': [(0.04, 0.12), 7], 'momentum': [(0.6, 0.8), 3]}]
### choose lr=0.07, momentum=0.7 for AGD
### first: 33.965 (+/-8.796), second: 37.500 (+/-9.347)
oct_fixed_noise = {'lr': 0.07, 'momentum':0.7, 'NCE': False, 'noise_eps': 0.1}
oct_noise_1 = [{'noise_r': [(0.02, 0.1), 3], 'noise_T': [(0, 100), 1]}] #200 runs, r=0.04, T=50
oct_noise_2 = [{'noise_r': [(0.01, 0.05), 3], 'noise_T': [(10, 60), 4]}] #200 runs, r=0.04, T=60
oct_noise_3 = [{'noise_r': [(0.001, 10), None], 'noise_T': [(0, 100), 4]}] #500 runs, r=1, T=0
oct_noise_4 = [{'noise_r': [(1, 6), 4], 'noise_T': [(0, 100), 4]}] #500 runs, r=3, T=0
### choose r=3, T=0, not stable
### first: 32.682 (+/-8.268), second: 34.014 (+/-7.756)
oct_fixed_NCE = {'lr': 0.07, 'momentum':0.7, 'NCE': True, 'noise_eps': 0.1, 'noise_r': 0}
oct_NCE_1 = [{'NCE_s': [(0.0001, 10), None]}] # s=1
oct_NCE_2 = [{'NCE_s': [(0.5, 4), 6]}] # s=3.5
oct_NCE_3 = [{'NCE_s': [(3, 6), 5]}] # s=4
oct_NCE_4 = [{'NCE_s': [(3, 5), 19]}] # s=4
### choose s=4, robust for s=3.5~5
### first: 22.470 (+/-3.528), second: 23.080 (+/-2.962)
oct_fixed_all = {'lr': 0.07, 'momentum':0.7, 'NCE': True, 'noise_eps': 0.1, 'noise_r': 3, 'noise_T': 0}
oct_all_1 = [{'NCE_s': [(3, 8), 4]}] # s=8
oct_all_2 = [{'NCE_s': [(7, 12), 4]}] # s=10
### choose s=10
### first: 18.828 (+/-4.231), second: 18.928 (+/-4.128)
#run(Octopus(), oct_all_2, fixed_params=oct_fixed_all, max_epochs=80, num_runs=500, metric='second', eps=0.1, verbose=True)


# Cosine function
cos_fixed_gd = {'noise_r': 0, 'momentum': 0, 'NCE': False, 'noise_eps': 0.1}
cos_gd_1 = [{'lr': [(0.0001, 10), None]}] #200 runs, best: lr=0.1
cos_gd_2 = [{'lr': [(0.5, 5), 8]}] #200 runs, best: lr=1, robust in 0.5~1.5
cos_gd_3 = [{'lr': [(0.5, 1.5), 9]}] #200 runs, best: lr=1.2, robust in 0.8~1.5
### choose lr=1.2
### first: 2.770 (+/-1.355), second: 2.855 (+/-1.508)
cos_fixed_gdnoise = {'lr': 1.2, 'momentum':0, 'NCE': False, 'noise_eps': 0.1}
cos_gdnoise_1 = [{'noise_r': [(0.01, 1), None], 'noise_T': [(0, 3), 2]}] #1000 runs, r=0.01, T=0, very unstable
cos_gdnoise_2 = [{'noise_r': [(0.05, 0.5), 8], 'noise_T': [(0, 1), 0]}] #1000 runs, r=0.1, T=1
### choose r=0.1, T=1
### first: 2.740 (+/-1.315), second: 2.774 (+/-1.363)
cos_fixed_agd = {'noise_r': 0, 'NCE': False, 'noise_eps': 0.1}
cos_agd_1 = [{'lr': [(0.5, 3), 4], 'momentum': [(0.1, 0.9), 7]}] #200 runs, best: lr=1, momentum=0.6
cos_agd_2 = [{'lr': [(0.6, 1.4), 3], 'momentum': [(0.3, 0.9), 5]}] #200 runs, best: lr=1.2, momentum=0.8
cos_agd_3 = [{'lr': [(1.1, 1.3), 1], 'momentum': [(0.6, 0.9), 2]}] #500 runs, best: lr=1.2, momentum=0.8
### choose lr=1.2, momentum=0.8
### first: 2.526 (+/-1.034), second: 2.586 (+/-1.104)
cos_fixed_noise = {'lr': 1.2, 'momentum':0.8, 'NCE': False, 'noise_eps': 0.1}
cos_noise_1 = [{'noise_r': [(0.01, 1), None], 'noise_T': [(0, 3), 2]}] #500 runs, r=0.1, T=0, second: 2.486 (+/-1.134)
cos_noise_2 = [{'noise_r': [(0.05, 0.5), 8], 'noise_T': [(0, 1), 0]}] #500 runs, r=0.1, T=0, not stable
### choose r=0.1, T=0
### first: 2.416 (+/-1.069), second: 2.486 (+/-1.134)
cos_fixed_NCE = {'lr': 1.2, 'momentum':0.8, 'NCE': True, 'noise_eps': 0.1, 'noise_r': 0}
cos_NCE_1 = [{'NCE_s': [(0.001, 10), None]}] # 200 runs, s=10, second: 2.610 (+/-1.130), no improvement
cos_NCE_2 = [{'NCE_s': [(2, 16), 6]}] # 200 runs, s=4, second: 2.425 (+/-0.935)
cos_NCE_3 = [{'NCE_s': [(3, 5), 1]}] # 1000 runs, s=3, second: 2.406 (+/-0.894)
### choose NCE_s=3
### first: 2.390 (+/-0.882), second: 2.406 (+/-0.894)
cos_fixed_all = {'lr': 1.2, 'momentum':0.8, 'NCE': True, 'noise_eps': 0.1, 'noise_r': 0.1, 'noise_T': 0}
cos_all_1 = [{'NCE_s': [(1, 15), 6]}] # 500 runs, s=3, second: 2.478 (+/-0.943)
cos_all_2 = [{'NCE_s': [(1, 5), 3]}] # 1000 runs, s=4, second: 2.421 (+/-1.001)
### choose NCE_s=4
### first: 2.392 (+/-0.960), second: 2.421 (+/-1.001)
#run(Cosine(), cos_gdnoise_2, fixed_params=cos_fixed_gdnoise, max_epochs=10, num_runs=1000, metric='second', eps=0.1, verbose=True)

#Quadratic function
qua_fixed_gd = {'noise_r': 0, 'momentum': 0, 'NCE': False, 'noise_eps': 0.1}
qua_gd_1 = [{'lr': [(0.0001, 10), None]}] #200 runs, best: lr=0.1
qua_gd_2 = [{'lr': [(0.05, 0.3), 4]}] #200 runs, best: lr=0.3
qua_gd_3 = [{'lr': [(0.3, 1.5), 5]}] #200 runs, best: lr=0.5
qua_gd_4 = [{'lr': [(0.4, 0.6), 3]}] #500 runs, best: lr=0.5
qua_fixed_all = {'lr': 0.5}
#run(Quadratic(), qua_gd_4, fixed_params=qua_fixed_gd, max_epochs=10, num_runs=500, metric='second', eps=0.1, verbose=True)
