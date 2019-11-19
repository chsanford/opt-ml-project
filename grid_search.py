import math
import time
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import numpy as np

from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, ParameterGrid, GridSearchCV as sk_GridSearchCV

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
    return data[0], data[1]
    #return data[0].numpy().astype(np.float32), data[1].numpy().astype(np.int64)


# Modeled after sklearn's GridSearchCV, but with pytorch modules
class GridSearchCV():
    scoring_options = ['accuracy', 'rmse']

    def __init__(self, model, optimizer, params, cv=2, verbose=True, max_epochs=20, scoring='accuracy', sgd=False, module_params=dict()):
        assert not sgd
        self.model_ = model
        self.optimizer_ = optimizer  # this is an uninstantiated class
        self.module_params = module_params
        self.param_grid = ParameterGrid(params)
        self.n_splits = cv
        self.max_epochs = max_epochs
        self.verbose = verbose
        if scoring not in GridSearchCV.scoring_options:
            raise Error(f'Unknown scoring {self.scoring}')
        self.scoring = scoring


    def fit(self, X, y):
        n_configs = len(self.param_grid)
        print(f'Ok, {self.n_splits} splits on {n_configs} configs, so {self.n_splits * n_configs} total fits.')
        self.cv_results_ = {'params': [], 'mean_test_score': [], 'std_test_score': []}
        kf = KFold(n_splits=self.n_splits)

        for i, params in enumerate(self.param_grid):
            print(f'[{i}/{n_configs}] {params}')
            scores = []

            for train_idx, test_idx in kf.split(X):
                model = self.model_(**self.module_params)
                optimizer = self.optimizer_(model.parameters(), is_ml=True, **params)
                scores.append(self._fit_and_score(model,
                                                  X, y,
                                                  optimizer,
                                                  train_idx,
                                                  test_idx))

            self.cv_results_['params'].append(params)
            self.cv_results_['mean_test_score'].append(np.mean(scores))
            self.cv_results_['std_test_score'].append(np.std(scores))

        rank_ = np.argmax if self.scoring == 'accuracy' else np.argmin
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
            train_loss = optimizer.step(closure).item()

            model.eval()
            with torch.no_grad():
                output = model(X[test_idx])
                valid_loss = loss_fn_(output, y[test_idx], reduction='mean')
                if self.scoring == 'accuracy':
                    pred = output.data.max(1, keepdim=True)[1]
                    score = pred.eq(y[test_idx].data.view_as(pred)).sum() / float(len(test_idx))
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
        return [min + i * step for i in range(num_imdt_values + 2)]
    else:  # log10 search
        min_exp = int(np.ceil(np.log10(min)))
        max_exp = int(np.floor(np.log10(max)))
        return [10 ** i for i in range(min_exp, max_exp + 1)]


# module = torch module, sk = whether we use sklearn's grid search, clf = true for classification or false for regression
# fixed_params is a dict of scalars, search_params is a dict ((min, max), num_imdt_values)
# set num_imdt_values >= 0 for linear search, None for log10 search
def grid_search(module_, X, y, cv=2, sk=False, clf=None, dims=None,
                fixed_params=dict(), search_params=dict(), max_epochs=20, sgd=False, verbose=True):
    prefix = 'optimizer__' if sk else ''
    optimizer = torch.optim.SGD if sgd else GradientDescent
    params = dict()

    for p, t in search_params.items():
        params[f'{prefix}{p}'] = interpolate(t[0][0], t[0][1], t[1])

    for p, v in fixed_params.items():
        params[f'{prefix}{p}'] = [v]

    if sk:
        skorch_wrapper_ = NeuralNetClassifier if clf else NeuralNetRegressor
        kwargs = {'module__log_prob': False} if clf else {'module__n': dims[0], 'module__m': dims[1], 'module__r': 20}
        scoring = 'accuracy' if clf else 'neg_mean_absolute_error'
        if not clf:
            X = X.astype(np.int64)
            y = y.reshape(-1, 1).astype(np.float32)

        sk_estimator = skorch_wrapper_(
            module_,
            optimizer=optimizer,
            max_epochs=max_epochs,
            batch_size=1 if sgd else -1,
            verbose=verbose,
            **kwargs,
        )

        gs = sk_GridSearchCV(sk_estimator, params, cv=cv, verbose=verbose, refit=False, scoring=scoring)
    else:
        kwargs = {'n': dims[0], 'm': dims[1], 'r': 20} if module_ == MatrixFactorization else {}
        scoring = 'accuracy' if clf else 'rmse'
        gs = GridSearchCV(module_, optimizer, params, cv=cv, verbose=verbose, max_epochs=max_epochs, scoring=scoring, sgd=sgd, module_params=kwargs)

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

    return {k[len(prefix):]: v for k, v in gs.best_params_.items()}


# Wrapper allows for sequential searches over the parameters
def run(module_, dataset, searches, fixed_params=dict(), verbose=True, sk=False):
    print(f'{module_} with {type(dataset).__name__}, running searches {searches} with fixed params {fixed_params}')

    max_epochs = 20
    X, y = load_dataset(dataset)
    is_mf = module_ == MatrixFactorization  # run sklearn grid search if mf, otherwise our own grid search
    dims = dataset.get_dims() if is_mf else None

    for i, search_params in enumerate(searches):
        s = f'Search [{i+1}/{len(searches)}] {search_params}'
        print(f"{''.ljust(len(s), '=')}\n{s}\n{''.ljust(len(s), '=')}\n")

        params = grid_search(module_, X, y, cv=2, sk=sk, clf=not is_mf, dims=dims,
                             fixed_params=fixed_params, search_params=search_params,
                             max_epochs=max_epochs, sgd=False, verbose=verbose)
        fixed_params.update(params)

    print(f'\nAll done, final parameters: {fixed_params}')


seq_searches = [{'lr': [(0.01, 0.1), 0]},
                {'momentum': [(0.01, 0.1), 0]}]

sim_searches = [{'lr': [(0.01, 0.1), 0],
                 'momentum': [(0.01, 0.1), 0]}]

run(FFNN, mnist_dataset, seq_searches, verbose=True, sk=False)

#run(MatrixFactorization, ml_dataset, seq_searches, verbose=True, sk=False)
