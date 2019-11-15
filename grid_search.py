import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import numpy as np

from torch.utils.data import DataLoader
from skorch import NeuralNetClassifier, NeuralNetRegressor
from sklearn.model_selection import GridSearchCV

from models.ffnn import FFNN
from models.cnn import CNN
from models.matrix_factorization import MatrixFactorization
from opt_methods.gradient_descent import GradientDescent
from datasets.MovieLens import MovieLensDataset


mnist_dataset = torchvision.datasets.MNIST(
    'data/mnist/train',
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
)

ml_dataset = MovieLensDataset(train=True)


def load_dataset(dataset):
    print('Loading dataset...')
    data = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    print('Finished loading the dataset into memory...')
    return data[0].numpy().astype(np.float32), data[1].numpy().astype(np.int64)


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


# module = torch module, clf = true for classification or false for regression
# fixed_params is a dict of scalars, search_params is a dict ((min, max), num_imdt_values)
# set num_imdt_values >= 0 for linear search, None for log10 search
def grid_search(module, X, y, clf,
                dims=None, fixed_params=dict(), search_params=dict(), max_epochs=20, sgd=False, verbose=True):

    sk_estimator_ = NeuralNetClassifier if clf else NeuralNetRegressor
    kwargs = {'module__log_prob': False} if clf else {'module__n': dims[0], 'module__m': dims[1], 'module__r': 20}
    params = dict()

    for p, t in search_params.items():
        params[f'optimizer__{p}'] = interpolate(t[0][0], t[0][1], t[1])

    for p, v in fixed_params.items():
        params[f'optimizer__{p}'] = [v]

    sk_estimator = sk_estimator_(
        module,
        #optimizer=torch.optim.SGD,
        optimizer=GradientDescent,
        max_epochs=max_epochs,
        batch_size=1 if sgd else -1,
        verbose=verbose,
        **kwargs,
    )

    gs = GridSearchCV(sk_estimator, params, cv=2, verbose=True, refit=False,
                      error_score='raise', scoring='accuracy' if clf else 'neg_mean_absolute_error')
    gs.fit(X, y)

    print("Best parameters set found on development set:")
    print(gs.best_params_)
    print()
    print("Grid scores on development set:")
    means = gs.cv_results_['mean_test_score']
    stds = gs.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, gs.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    return {k[len('optimizer__'):]: v for k, v in gs.best_params_.items()}  # strip skorch prefix


# Wrapper allows for sequential searches over the parameters
def run(module, dataset, searches, fixed_params=dict(), verbose=True):
    print(f'{module} with {type(dataset).__name__}, running searches {searches} with fixed params {fixed_params}')

    max_epochs = 20
    X, y = load_dataset(dataset)
    clf = module != MatrixFactorization
    dims = dataset.get_dims() if not clf else None

    for i, search_params in enumerate(searches):
        s = f'Search [{i+1}/{len(searches)}] {search_params}'
        print(f"{''.ljust(len(s), '=')}\n{s}\n{''.ljust(len(s), '=')}\n")

        params = grid_search(module, X, y, clf, dims=dims,
                             fixed_params=fixed_params, search_params=search_params,
                             max_epochs=max_epochs, sgd=False, verbose=verbose)
        fixed_params.update(params)

    print(f'\nAll done, final parameters: {fixed_params}')


seq_searches = [{'lr': [(0.01, 0.1), 0]},
                {'momentum': [(0.01, 0.1), 0]}]

sim_searches = [{'lr': [(0.01, 0.1), 0],
                 'momentum': [(0.01, 0.1), 0]}]

run(FFNN, mnist_dataset, sim_searches, verbose=True)
