import math
import time
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import numpy as np

from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, ParameterGrid

from models.ffnn import FFNN
from models.cnn import CNN
from models.matrix_factorization import MatrixFactorization
from ml_testing.mf_testing import MatrixFactorizationTest
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
    return data[0], data[1]


# Modeled after sklearn's GridSearchCV, but with pytorch modules
class GridSearchCV():
    scoring_options = ['accuracy', 'rmse']

    def __init__(self, model_, optimizer_, params, cv=2, verbose=True, max_epochs=20, scoring='accuracy', sgd=False, module_params=dict(), initial_state=None, all_folds=False, train_score=True):
        assert not sgd  # not supported right now
        self.model_ = model_
        self.optimizer_ = optimizer_
        self.module_params = module_params
        self.param_grid = ParameterGrid(params)
        self.n_splits = cv
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.initial_state = initial_state
        self.all_folds = all_folds  # whether to perform CV or just use one split
        self.train_score = train_score  # whether to score based on train or valid loss
        if scoring not in GridSearchCV.scoring_options:
            raise Error(f'Unknown scoring {self.scoring}')
        self.scoring = scoring
        if self.verbose:
            print(params)


    # Drop test indices that weren't in the training set if MF
    def _check_mf(self, X, train_idx, test_idx):
        if self.model_ == MatrixFactorization:
            test_idx = test_idx[np.in1d(X[test_idx][:, 0], X[train_idx][:, 0])]
            test_idx = test_idx[np.in1d(X[test_idx][:, 1], X[train_idx][:, 1])]
        return test_idx


    def fit(self, X, y):
        n_configs = len(self.param_grid)
        print(f'Ok, {self.n_splits} splits on {n_configs} configs, so {self.n_splits * n_configs} total fits. Size of fold: {int(len(X) / self.n_splits)}')
        self.cv_results_ = {'params': [], 'mean_test_score': [], 'std_test_score': []}
        kf = KFold(n_splits=self.n_splits, shuffle=False)

        for i, params in enumerate(self.param_grid):
            print(f'[{i+1}/{n_configs}] {params}')
            scores = []

            for train_idx, test_idx in kf.split(X):
                if self.train_score:  # use all the data for training and scoring
                    train_idx = range(len(X))
                    test_idx = []
                else:
                    test_idx = self._check_mf(X, train_idx, test_idx)

                model = self.model_(**self.module_params)

                if self.initial_state is not None:
                    model.load_state_dict(self.initial_state, strict=True)

                if self.optimizer_ == GradientDescent:
                    extra_params = {'is_ml': True}
                else:
                    extra_params = {'nesterov': True}

                optimizer = self.optimizer_(model.parameters(), **params, **extra_params)
                scores.append(self._fit_and_score(model,
                                                  X, y,
                                                  optimizer,
                                                  train_idx,
                                                  test_idx))

                if not self.all_folds or self.train_score:
                    break

            self.cv_results_['params'].append(params)
            self.cv_results_['mean_test_score'].append(np.mean(scores))
            self.cv_results_['std_test_score'].append(np.std(scores))

        rank_ = np.nanargmax if self.scoring == 'accuracy' else np.nanargmin
        self.best_params_ = self.cv_results_['params'][rank_(self.cv_results_['mean_test_score'])]


    def _fit_and_score(self, model, X, y, optimizer, train_idx, test_idx):
        loss_fn_ = F.mse_loss if self.scoring == 'rmse' else F.nll_loss
        def closure():
            optimizer.zero_grad()
            output = model(X[train_idx])
            loss = loss_fn_(output, y[train_idx])
            loss.backward()
            return loss

        for epoch in range(self.max_epochs):
            start = time.time()
            model.train()
            train_loss = optimizer.step(closure).item()

            if not np.isfinite(train_loss):
                valid_loss = np.nan
                break

            if self.train_score:
                valid_loss = train_loss
                score = train_loss
            else:
                model.eval()
                with torch.no_grad():
                    idx = train_idx if self.train_score else test_idx
                    output = model(X[idx])
                    valid_loss = loss_fn_(output, y[idx])
                    if not np.isfinite(valid_loss):
                        valid_loss = np.nan
                        break
                    if self.scoring == 'accuracy':
                        pred = output.data.max(1, keepdim=True)[1]
                        score = pred.eq(y[idx].data.view_as(pred)).sum() / float(len(idx))
                    elif self.scoring == 'rmse':
                        score = math.sqrt(valid_loss)

            if self.verbose:
                self._log(epoch, train_loss, valid_loss, time.time() - start)

        return score

    def _log(self, epoch, train_loss, valid_loss, dur):
        if epoch == 0:
            print('{:>5s} {:>10s} {:>10s} {:>7s}'.format('epoch', 't_loss', 'v_loss', 'dur'))
            print('=' * 35)
        print(f'{epoch+1:5d} {train_loss:10.3f} {valid_loss:10.3f} {dur:7.2f}')
        if epoch == self.max_epochs - 1:
            print()


# linear is straightforward; if log10, then min and max will be chosen to be the closest exp. of 10 (e.g. [0.0005, 2] -> [0.001, 0.01, 0.1, 1])
def interpolate(min, max, num_imdt_values):
    assert min < max
    if num_imdt_values is not None:  # linear search
        assert num_imdt_values >= 0
        step = (max - min) / (num_imdt_values + 1)
        return [round(min + i * step, 5) for i in range(num_imdt_values + 2)]
    else:  # log10 search
        min_exp = int(np.ceil(np.log10(min)))
        max_exp = int(np.floor(np.log10(max)))
        return [10 ** i for i in range(min_exp, max_exp + 1)]


# fixed_params is a dict of scalars, search_params is a dict ((min, max), num_imdt_values)
# set num_imdt_values >= 0 for linear search, None for log10 search
def grid_search(module_, X, y, cv=2, dims=None,
                fixed_params=dict(), search_params=dict(), max_epochs=20, sgd=False, verbose=True,
                initial_state=None):
    optimizer = torch.optim.SGD if sgd else GradientDescent
    params = dict()

    for p, t in search_params.items():
        params[p] = interpolate(t[0][0], t[0][1], t[1])

    for p, v in fixed_params.items():
        params[p] = [v]

    if module_ == MatrixFactorization:
        kwargs = {'n': dims[0], 'm': dims[1], 'r': 20}
        scoring = 'rmse'
    else:
        kwargs = {}
        scoring = 'accuracy'

    gs = GridSearchCV(module_, optimizer, params,
                      cv=cv, verbose=verbose, max_epochs=max_epochs,
                      scoring=scoring, sgd=sgd, module_params=kwargs, initial_state=initial_state)
    gs.fit(X, y)

    print(f'Best parameters set found on development set:\n{gs.best_params_}')
    print(f'\nGrid scores on development set ({scoring}):')
    means = gs.cv_results_['mean_test_score']
    stds = gs.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, gs.cv_results_['params']):
        print(f'{mean:0.6f} (+/-{std * 2:0.03f}) for {params}')

    return gs.best_params_.items()


# Wrapper allows for sequential searches over the parameters
def run(module_, dataset, searches, fixed_params=dict(), cv=2, max_epochs=20, verbose=True, initial_state=None):
    print(f'{module_} with {type(dataset).__name__}, running searches {searches} with fixed params {fixed_params}\n')
    if initial_state is not None:
        print('Running with pre-trained parameters.')

    X, y = load_dataset(dataset)
    dims = dataset.get_dims() if module_ == MatrixFactorization else None

    for i, search_params in enumerate(searches):
        s = f'Search [{i+1}/{len(searches)}] {search_params}'
        print(f"{''.ljust(len(s), '=')}\n{s}\n{''.ljust(len(s), '=')}\n")

        params = grid_search(module_, X, y, cv=cv, dims=dims,
                             fixed_params=fixed_params, search_params=search_params,
                             max_epochs=max_epochs, sgd=False, verbose=verbose, initial_state=initial_state)
        fixed_params.update(params)

    print(f'\nAll done, final parameters: {fixed_params}')


seq_searches = [{'lr': [(0.001, 10), None]},
                {'momentum': [(0.01, 0.1), 0]}]

sim_searches = [{'lr': [(0.01, 100), None],
                 'momentum': [(0.01, 0.1), 0]}]

mf_gd = [{'lr': [(10, 60), 1]}]
mf_gd2 = [{'lr': [(30, 38), 3]}]
mf_agd = [{'lr': [(10, 50), 3],
           'momentum': [(0.1, 0.9), 1]}]
mf_agd2 = [{'lr': [(16, 22), 2],
           'momentum': [(0.9, 0.95), 0]}]
mf_noise = [{'noise_r': [(0.1, 10), None],
                'noise_eps': [(0.1, 10), None],
                'noise_T': [(1, 15), 2]}]
mf_noise2 = [{'noise_r': [(0.025, 0.1), 2],
                 'noise_eps': [(0.1, 1), 0]}]
mf_nce = [{'NCE_gamma': [(0.01, 100), None],
           'NCE_s': [(0.01, 100), None]}]
mf_test = [{'NCE_s': [(1, 10), None]}]

max_epochs = 200
cv = 8
state_dict = torch.load(MatrixFactorizationTest.path)

run(MatrixFactorization, ml_dataset,
    mf_nce,
    fixed_params={'lr': 22, 'momentum': 0.95,
                  'noise_T': 1, 'noise_r': 0.1, 'noise_eps': 0.1,
                  'NCE': True, 'NCE_gamma': 0},
    initial_state=state_dict,
    max_epochs=max_epochs, cv=cv, verbose=True)
