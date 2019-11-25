import math
import os
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


"""
Top-level script that runs a grid search on the torch models, printing the results to console.
The parameter ranges we tested are at the bottom.
"""

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
log = False


# Optionally print to a log file.
def print_(s, filename='log/log.txt'):
    print(s)
    if log:
        with open(filename, 'a') as fp:
            print(s, file=fp)


# Loads all the data into memory for faster training.
def load_dataset(dataset):
    print_('Loading dataset...')
    data = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    print_('Finished loading the dataset into memory...')
    #print(f'# of data: {len(data[0])}')
    return data[0], data[1]


# Modeled after sklearn's GridSearchCV, but with pytorch modules.
class GridSearchCV():
    scoring_options = ['accuracy', 'rmse']

    def __init__(self, model_, optimizer_, params, cv=2, verbose=True, max_epochs=20, scoring='accuracy', sgd=False, module_params=dict(), initial_state=None, all_folds=False, train_score=True, data=-1, num_run=1):
        assert not sgd  # not supported right now
        self.model_ = model_  # instantiates later
        self.optimizer_ = optimizer_  # instantiates later
        self.module_params = module_params
        self.param_grid = ParameterGrid(params)
        self.n_splits = cv
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.initial_state = initial_state
        self.all_folds = all_folds  # whether to perform CV or just use one split
        self.train_score = train_score  # whether to score based on train or valid loss
        self.num_run = num_run
        self.data = data  # how many data points to use, if not the full training data
        if scoring not in GridSearchCV.scoring_options:
            raise Error(f'Unknown scoring {self.scoring}')
        self.scoring = scoring
        if self.verbose:
            print_(params)


    # Drop test indices that weren't in the training set if MF.
    def _check_mf(self, X, train_idx, test_idx):
        if self.model_ == MatrixFactorization:
            test_idx = test_idx[np.in1d(X[test_idx][:, 0], X[train_idx][:, 0])]
            test_idx = test_idx[np.in1d(X[test_idx][:, 1], X[train_idx][:, 1])]
        return test_idx


    # Runs the grid search on all parameter configurations.
    def fit(self, X, y):
        n_configs = len(self.param_grid)
        print_(f'Ok, {self.n_splits} splits on {n_configs} configs, so {self.n_splits * n_configs} total fits. Size of fold: {int(len(X) / self.n_splits)}')
        self.cv_results_ = {'params': [], 'mean_test_score': [], 'std_test_score': []}
        kf = KFold(n_splits=self.n_splits, shuffle=False)

        for i, params in enumerate(self.param_grid):
            print_(f'[{i+1}/{n_configs}] {params}')
            scores = []

            for i in range(self.num_run):
                for train_idx, test_idx in kf.split(X):
                    if self.train_score:  # use all the data for training and scoring
                        if self.data > 0:
                            train_idx = range(self.data)
                        else:
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
                    score = self._fit_and_score(model, X, y, optimizer, train_idx, test_idx)
                    scores.append(score)

                    if not self.all_folds or self.train_score:
                        break
            self.cv_results_['params'].append(params)
            self.cv_results_['mean_test_score'].append(np.mean(scores))
            self.cv_results_['std_test_score'].append(np.std(scores))

        rank_ = np.nanargmax if (self.scoring == 'accuracy' and not self.train_score) else np.nanargmin
        self.best_params_ = self.cv_results_['params'][rank_(self.cv_results_['mean_test_score'])]


    # Perform training and scoring.
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
            print_('{:>5s} {:>10s} {:>10s} {:>7s}'.format('epoch', 't_loss', 'v_loss', 'dur'))
            print_('=' * 35)
        print_(f'{epoch+1:5d} {train_loss:10.3f} {valid_loss:10.3f} {dur:7.2f}')
        if epoch == self.max_epochs - 1:
            print_('')


# Generates the parameter ranges.
# Linear is straightforward; if log10, then min and max will be chosen to be the closest exp. of 10
# (e.g. [0.0005, 2] -> [0.001, 0.01, 0.1, 1])
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


# Wrapper that runs a GridSearchCV and reports the results.
# fixed_params: dict of scalars, search_params: dict ((min, max), num_imdt_values)
# set num_imdt_values >= 0 for linear search, None for log10 search
def grid_search(module_, X, y, cv=2, dims=None,
                fixed_params=dict(), search_params=dict(), max_epochs=20, sgd=False, verbose=True,
                initial_state=None, data=-1, num_run=1):
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
                      scoring=scoring, sgd=sgd, module_params=kwargs, initial_state=initial_state,
                      data=data, num_run=num_run)
    gs.fit(X, y)

    print_(f'Best parameters set found on development set:\n{gs.best_params_}')
    print_(f'\nGrid scores on development set ({scoring}):')
    means = gs.cv_results_['mean_test_score']
    stds = gs.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, gs.cv_results_['params']):
        print_(f'{mean:0.6f} (+/-{std * 2:0.03f}) for {params}')

    return gs.best_params_.items()


# Wrapper allows for sequential searches over the parameters
def run(module_, dataset, searches, fixed_params=dict(), cv=2, max_epochs=20, verbose=True, initial_state=None, data=-1, num_run=1):
    print_(f'{module_} with {type(dataset).__name__}, running searches {searches} with fixed params {fixed_params}')
    print_(f'Average over {num_run} runs.\n')
    if initial_state is not None:
        print_('Running with pre-trained parameters.')

    X, y = load_dataset(dataset)
    dims = dataset.get_dims() if module_ == MatrixFactorization else None

    for i, search_params in enumerate(searches):
        s = f'Search [{i+1}/{len(searches)}] {search_params}'
        print_(f"{''.ljust(len(s), '=')}\n{s}\n{''.ljust(len(s), '=')}\n")

        params = grid_search(module_, X, y, cv=cv, dims=dims,
                             fixed_params=fixed_params, search_params=search_params,
                             max_epochs=max_epochs, sgd=False, verbose=verbose, initial_state=initial_state,
                             data=data, num_run=num_run)
        fixed_params.update(params)

    print_(f'\nAll done, final parameters: {fixed_params}')
    print_('')


# Pre-train a MF model for 200 epochs using vanilla GD to speed up grid search.
def get_pretrain_state():
    if not os.path.exists(MatrixFactorizationTest.path):
        mf_test = MatrixFactorizationTest(load_model=False)
        mf_optim = GradientDescent(mf_test.model.parameters(),
                                   is_ml=True,
                                   lr=10)
        mf_test.run(200, mf_optim, sgd=False, save_model=True, log=False)

    return torch.load(MatrixFactorizationTest.path)


# Configure these!
max_epochs = 200
cv = 8

# ====================
# MATRIX FACTORIZATION
# ====================

mf_gd = [{'lr': [(10, 60), 4]}]
mf_gd2 = [{'lr': [(42, 58), 3]}]
# best: lr = 36

mf_agd = [{'lr': [(10, 50), 3],
           'momentum': [(0.1, 0.9), 1]}]
mf_agd2 = [{'lr': [(16, 30), 6],
            'momentum': [(0.85, 0.95), 1]}]
# best: lr = 22, momentum = 0.95

mf_noise = [{'noise_r': [(0.1, 1), 1],
                'noise_eps': [(0.1, 10), None],
                'noise_T': [(1, 15), 2]}]
mf_noise2 = [{'noise_r': [(0.025, 0.1), 2],
                 'noise_eps': [(0.1, 1), 0]}]
# best (vanilla): noise_T = 1, noise_eps = 0.1, noise_r = 1 (minimal effect)
# best (AGD): noise_T = 1, noise_eps = 0.1, noise_r = 0.1 (minimal effect)

mf_nce_test = [{'NCE_s': [(1, 10), None]}]
# best: NCE never invoked.

state_dict = get_pretrain_state()
# The following two lines reproduce our grid search.
# Vanilla GD & Vanilla GD with Noise
run(MatrixFactorization, ml_dataset,
    [mf_gd[0], mf_noise2[0]],
    fixed_params=dict(),
    initial_state=state_dict,
    max_epochs=max_epochs, cv=cv, verbose=True)

# AGD & AGD with Noise & AGD with Noise and NCE
run(MatrixFactorization, ml_dataset,
    [mf_agd[2], mf_noise2[0], mf_nce_test],
    fixed_params=dict(),
    initial_state=state_dict,
    max_epochs=max_epochs, cv=cv, verbose=True)

# ==============================
# FULLY CONNECTED NEURAL NETWORK
# ==============================
fnn_gd1 = [{'lr': [(0.05, 0.4), 6]}]  # best: 0.25
fnn_gd2 = [{'lr': [(0.2, 0.3), 4]}]  # best: 0.3, but not stable
run(FFNN, mnist_dataset, fnn_gd1, fixed_params={'momentum': 0, 'noise_r': 0, 'NCE': False}, max_epochs=max_epochs, verbose=True, num_run=5)
run(FFNN, mnist_dataset, fnn_gd2, fixed_params={'momentum': 0, 'noise_r': 0, 'NCE': False}, max_epochs=max_epochs, verbose=True, num_run=5)
# [GD] lr=0.25, loss~=0.28

fnn_agd1 = [{'lr': [(0.05, 0.4), 6], 'momentum': [(0.1, 0.9), 7]}]  # best: lr=0.15, momentum=0.9
fnn_agd2 = [{'lr': [(0.12, 0.3), 8], 'momentum': [(0.9, 0.95), 0]}]  # best: lr=0.18, momentum=0.95, quite robust for lr=0.12~0.2
run(FFNN, mnist_dataset, fnn_agd1, fixed_params={'noise_r': 0, 'NCE': False}, max_epochs=max_epochs, verbose=True, num_run=5)
run(FFNN, mnist_dataset, fnn_agd2, fixed_params={'noise_r': 0, 'NCE': False}, max_epochs=max_epochs, verbose=True, num_run=5)
# [AGD] lr=0.18, momentum=0.95, loss~=0.24

fnn_noise1 = [{'noise_r': [(0.001, 10), None], 'noise_T': [(0, 100), 1], 'noise_eps': [(0.001, 10), None]}] #best: T=100, eps=1, r=0.01, but only runned for 1 time, so result not stable
fnn_noise2 = [{'noise_r': [(0.01, 0.1), None], 'noise_T': [(0, 100), 1]}] #fix eps=0.1, best: T=100, r=0.1
fnn_noise3 = [{'noise_r': [(0.05, 0.2), 2], 'noise_T': [(50, 100), 0]}] #fix eps=0.1, run 10 times, best: T=50, r=0.1
run(FFNN, mnist_dataset, fnn_noise1, fixed_params={'lr': 0.18, 'momentum': 0.95, 'NCE': False}, max_epochs=max_epochs, verbose=True, num_run=1)
run(FFNN, mnist_dataset, fnn_noise2, fixed_params={'lr': 0.18, 'momentum': 0.95, 'NCE': False}, max_epochs=max_epochs, verbose=True, num_run=5)
run(FFNN, mnist_dataset, fnn_noise3, fixed_params={'lr': 0.18, 'momentum': 0.95, 'NCE': False}, max_epochs=max_epochs, verbose=True, num_run=5)
# [AGD + noise] noise_r=0.1, noise_eps=0.1, noise_T=50, loss = 0.242244 (+/-0.005)

fnn_NCE1 = [{'NCE_s': [(0.01, 10), None]}] #1 run, best: s=10, (0.1, 1 also get similar loss)
fnn_NCE2 = [{'NCE_s': [(2, 8), 2]}] #5 runs, best: s=6, loss=0.240498 (+/-0.004)
fnn_NCE3 = [{'NCE_s': [(5, 7), 1]}] #10 runs, best: s=7
run(FFNN, mnist_dataset, fnn_NCE1, fixed_params={'lr': 0.18, 'momentum': 0.95, 'NCE': True, 'noise_r': 0}, max_epochs=max_epochs, verbose=True, num_run=1)
run(FFNN, mnist_dataset, fnn_NCE2, fixed_params={'lr': 0.18, 'momentum': 0.95, 'NCE': True, 'noise_r': 0}, max_epochs=max_epochs, verbose=True, num_run=5)
run(FFNN, mnist_dataset, fnn_NCE3, fixed_params={'lr': 0.18, 'momentum': 0.95, 'NCE': True, 'noise_r': 0}, max_epochs=max_epochs, verbose=True, num_run=10)
# [AGD + NCE] NCE_s=7, loss = 0.241532 (+/-0.005)

fnn_all1 = [{'NCE_s': [(2, 8), 2]}] #5 runs, best: s=4, loss=0.241248 (+/-0.004)
run(FFNN, mnist_dataset, fnn_all1, fixed_params={'lr': 0.18, 'momentum': 0.95, 'noise_r': 0.1, 'noise_eps': 0.1, 'noise_T': 50}, max_epochs=max_epochs, verbose=True, num_run=5)
# [all] NCE_s=4, loss=0.241248 (+/-0.004)



# ============================
# CONVOLUTIONAL NEURAL NETWORK
# ============================
cnn_gd1 = [{'lr': [(0.01, 10), None]}] #best: 0.1 
cnn_gd2 = [{'lr': [(0.05, 0.3), 4]}] #best: 0.2, loss=0.030520
cnn_gd3 = [{'lr': [(0.12, 0.26), 6]}] #best: 0.26, loss=0.009335, but not stable
run(CNN, mnist_dataset, cnn_gd1, fixed_params={'momentum': 0,'noise_r': 0}, max_epochs=max_epochs, verbose=True, data=3000)
run(CNN, mnist_dataset, cnn_gd2, fixed_params={'momentum': 0,'noise_r': 0}, max_epochs=max_epochs, verbose=True, data=3000)
run(CNN, mnist_dataset, cnn_gd3, fixed_params={'momentum': 0,'noise_r': 0}, max_epochs=max_epochs, verbose=True, data=3000)
# [GD] lr=0.26

cnn_agd1 = [{'lr': [(0.05, 0.4), 6], 'momentum': [(0.3, 0.9), 1]}] #best: lr=0.05, momentum=0.9, loss=0.016273
run(CNN, mnist_dataset, cnn_agd1, fixed_params={'noise_r': 0}, max_epochs=max_epochs, verbose=True, data=3000)
# [AGD] lr=0.05, momentum=0.9

cnn_noise1 = [{'noise_r': [(0.01, 1), None], 'noise_T': [(0, 100), 1]}] #2 runs, best: r=1, T=50
run(CNN, mnist_dataset, cnn_noise1, fixed_params={'lr': 0.05, 'momentum': 0.9}, max_epochs=200, verbose=True, data=3000, num_run=2)
# [AGD + noise] noise_r=1, noise_T=50

cnn_nce1 = [{'NCE_s': [(2,8), 2]}] #3 runs, best: s=6, loss=0.002380 (+/-0.001)
run(CNN, mnist_dataset, cnn_nce1, fixed_params={'lr': 0.05, 'momentum': 0.9, 'noise_r': 0}, max_epochs=200, verbose=True, data=3000, num_run=3)
# [AGD + NCE] NCE_s=6

