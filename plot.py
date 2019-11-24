import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch, torchvision

from datasets.MovieLens import MovieLensDataset
from opt_methods.gradient_descent import GradientDescent

from simple_testing.quadratic_function import Quadratic
from simple_testing.cosine_function import Cosine
from simple_testing.octopus_function import Octopus
from simple_testing.simple_testing import run_trials

from ml_testing.mf_testing import MatrixFactorizationTest
from ml_testing.mnist_testing import MnistTest 

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



def print_(s, filename='log/data.txt'):
    print(s)
    with open(filename, 'a') as fp:
        print(s, file=fp)

def load_logs(paths):
    logs = []
    
    for i, fname in enumerate(paths):
        with open(fname, 'rb') as f:
            d = pickle.load(f)
            logs.append(d)
            print(f'[{i}] {d["name"]}\n{d["params"]}')
    return logs


def save_plot(fig, path):
    fig.savefig(path, dpi=300)

def make_loss_plot(ax, data, labels, ylabel, ylim=None):
    ax.margins(0)
    ax.set_xlabel('epochs')
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(top=ylim) 
        
    for i, d in enumerate(data):
        ax.plot(d, label=labels[i])
        
    ax.legend()
    return ax

def make_histogram(ax, epochs, data, labels, name):
    bins = np.linspace(0, epochs, epochs)
    ax.hist(data, bins, label=labels)
    ax.legend(loc='upper right')
    ax.set_xlabel(name)
    ax.set_ylabel('number of trials')

'''
# ================
# NON-ML FUNCTIONS
# ================
# Octopus optimizers
oct_optimizer = dict()
oct_optimizer['gd'] = GradientDescent(None, is_ml=False, lr=0.13, momentum=0, noise_r=0, NCE=False, is_verbose=False)
oct_optimizer['gd_noise'] = GradientDescent(None, is_ml=False, lr=0.13, momentum=0, noise_r=5, noise_T=0, noise_eps=0.1, NCE=False, is_verbose=False)
oct_optimizer['agd'] = GradientDescent(None, is_ml=False, lr=0.07, momentum=0.7, noise_r=0, NCE=False, is_verbose=False)
oct_optimizer['agd_noise'] = GradientDescent(None, is_ml=False, lr=0.07, momentum=0.7, noise_r=3, noise_T=0, noise_eps=0.1, NCE=False, is_verbose=False)
oct_optimizer['agd_NCE'] = GradientDescent(None, is_ml=False, lr=0.07, momentum=0.7, noise_r=0, NCE=True, NCE_s=4, is_verbose=False)
oct_optimizer['all'] = GradientDescent(None, is_ml=False, lr=0.07, momentum=0.7, noise_r=3, noise_T=0, noise_eps=0.1, NCE=True, NCE_s=10, is_verbose=False)

# Cosine optimizers
cos_optimizer = dict()
cos_optimizer['gd'] = GradientDescent(None, is_ml=False, lr=1.2, momentum=0, noise_r=0, NCE=False, is_verbose=False)
cos_optimizer['gd_noise'] = GradientDescent(None, is_ml=False, lr=1.2, momentum=0, noise_r=0.1, noise_T=1, noise_eps=0.1, NCE=False, is_verbose=False)
cos_optimizer['agd'] = GradientDescent(None, is_ml=False, lr=1.2, momentum=0.8, noise_r=0, NCE=False, is_verbose=False)
cos_optimizer['agd_noise'] = GradientDescent(None, is_ml=False, lr=1.2, momentum=0.8, noise_r=0.1, noise_T=0, noise_eps=0.1, NCE=False, is_verbose=False)
cos_optimizer['agd_NCE'] = GradientDescent(None, is_ml=False, lr=1.2, momentum=0.8, noise_r=0, NCE=True, NCE_s=3, is_verbose=False)
cos_optimizer['all'] = GradientDescent(None, is_ml=False, lr=1.2, momentum=0.8, noise_r=0.1, noise_T=0, noise_eps=0.1, NCE=True, NCE_s=4, is_verbose=False)

# Quadratic optimizers
#qua_optimizer = dict()
#qua_optimizer['gd'] = GradientDescent(None, is_ml=False, lr=0.5, momentum=0, noise_r=0, NCE=False, is_verbose=False)

# run experiments
trials = 10000
epochs=dict()
epochs['Octopus'] = 150
epochs['Cosine'] = 10
oct_filenames = dict()
cos_filenames = dict()
for name, optimizer in oct_optimizer.items():
    oct_filenames[name] = run_trials(Octopus(), optimizer, trials=trials, epochs=epochs['Octopus'], tag=name)
#for name, optimizer in cos_optimizer.items():
    cos_filenames[name] = run_trials(Cosine(), optimizer, trials=trials, epochs=epochs['Cosine'], tag=name)

#oct_filenames = {'gd': './log/Octopus-2-11-23T12:55:53-90', 'gd_noise': './log/Octopus-2-11-23T12/57/04-52', 'agd': './log/Octopus-2-11-23T12/57/38-59', 'agd_noise': './log/Octopus-2-11-23T12/59/11-26', 'agd_NCE': './log/Octopus-2-11-23T13/00/56-66', 'all': './log/Octopus-2-11-23T13:02:58-51'}
# save average data
for fname, filenames in {'Octopus': oct_filenames, 'Cosine': cos_filenames}.items():
    print_('==================')
    print_(fname)
    print_(f'epochs = {epochs[fname]}')
    print_(f'trials = {trials}')
    print_('==================')
    for name, filename in filenames.items():
        logs = load_logs([filename])[0]
        print_(f'optimizer [{name}]')
        print_(f'{logs["params"]}')
        mean_fosp = np.mean(logs['fosp'])
        std_fosp = np.std(logs['fosp'])
        mean_sosp = np.mean(logs['sosp'])
        std_sosp = np.std(logs['sosp'])
        print_(f'mean_fosp = {mean_fosp}')
        print_(f'std_fosp = {std_fosp}')
        print_(f'mean_sosp = {mean_sosp}')
        print_(f'std_sosp = {std_sosp}')
# make histogram
logs = load_logs([oct_filenames['gd'], oct_filenames['all']])
fig, ax = plt.subplots()
make_histogram(ax, epochs['Octopus'], [logs[0]['sosp'], logs[1]['sosp']], ['Vanilla Gradient Descent', 'Perturbed Accelerated Gradient Descent'], 'epochs to second order stationary point')
fig.savefig("oct_hist.png", dpi=600)

# make loss graph
fig, ax = plt.subplots()
make_loss_plot(ax, [np.mean(logs[0]['losses'], axis=0), np.mean(logs[1]['losses'], axis=0)], ['Vanilla Gradient Descent', 'Perturbed Accelerated Gradient Descent'], 'objective value')
fig.savefig("oct_loss.png", dpi=600)
'''

# ================
# ML FUNCTIONS
# ================
'''
# FNN optimizers
fnn_test = MnistTest(ff=True)
fnn_optimizer = dict()
fnn_optimizer['gd'] = GradientDescent(fnn_test.model.parameters(), is_ml=True, lr=0.25, momentum=0, noise_r=0, NCE=False, is_verbose=False)
fnn_optimizer['agd'] = GradientDescent(fnn_test.model.parameters(), is_ml=True, lr=0.18, momentum=0.95, noise_r=0, NCE=False, is_verbose=False)
fnn_optimizer['agd_noise'] = GradientDescent(fnn_test.model.parameters(), is_ml=True, lr=0.18, momentum=0.95, noise_r=0.1, noise_T=50, noise_eps=0.1, NCE=False, is_verbose=False)
fnn_optimizer['agd_NCE'] = GradientDescent(fnn_test.model.parameters(), is_ml=True, lr=0.18, momentum=0.95, noise_r=0, noise_T=50, noise_eps=0.1, NCE=True, NCE_s=7, is_verbose=False)
fnn_optimizer['all'] = GradientDescent(fnn_test.model.parameters(), is_ml=True, lr=0.18, momentum=0.95, noise_r=0.1, noise_T=50, noise_eps=0.1, NCE=True, NCE_s=4, is_verbose=False)

# CNN optimizers
cnn_test = MnistTest(ff=False)
cnn_optimizer = dict()
cnn_optimizer['gd'] = GradientDescent(cnn_test.model.parameters(), is_ml=True, lr=0.26, momentum=0, noise_r=0, NCE=False, is_verbose=False)
cnn_optimizer['agd'] = GradientDescent(cnn_test.model.parameters(), is_ml=True, lr=0.05, momentum=0.9, noise_r=0, NCE=False, is_verbose=False)
cnn_optimizer['agd_noise'] = GradientDescent(cnn_test.model.parameters(), is_ml=True, lr=0.05, momentum=0.9, noise_r=1, noise_T=50, NCE=False, is_verbose=False)
cnn_optimizer['agd_NCE'] = GradientDescent(cnn_test.model.parameters(), is_ml=True, lr=0.05, momentum=0.9, noise_r=0, NCE=True, NCE_s=6, is_verbose=False)
cnn_optimizer['all'] = GradientDescent(cnn_test.model.parameters(), is_ml=True, lr=0.05, momentum=0.9, noise_r=1, noise_T=50, NCE=True, NCE_s=6, is_verbose=False)
'''

# Matrix Factorization optimizers
mf_test = MatrixFactorizationTest(load_model=False)
mf_optimizer = dict()
#mf_optimizer['gd'] = GradientDescent(mf_test.model.parameters(), is_ml=True, lr=36, momentum=0, noise_r=0, NCE=False, is_verbose=False)
#mf_optimizer['gd_noise'] = GradientDescent(mf_test.model.parameters(), is_ml=True, lr=36, momentum=0, noise_r=1, noise_T=1, noise_eps=0.1, NCE=False, is_verbose=False)
mf_optimizer['agd'] = GradientDescent(mf_test.model.parameters(), is_ml=True, lr=22, momentum=0.95, noise_r=0, NCE=False, is_verbose=False)
mf_optimizer['all'] = GradientDescent(mf_test.model.parameters(), is_ml=True, lr=22, momentum=0.95, noise_r=1, noise_T=1, noise_eps=0.1, NCE=False, is_verbose=False)


# run experiments
trials=dict()
epochs=dict()
trials['fnn'] = 10
trials['cnn'] = 3
trials['mf'] = 10
epochs['fnn'] = 200
epochs['cnn'] = 200
epochs['mf'] = 200
#all_filenames={'fnn':dict(), 'cnn':dict()}
all_filenames={'fnn':dict(), 'cnn':dict(), 'mf':dict()}
'''
for name, optimizer in fnn_optimizer.items():
    print(f'Train FNN using optimizer {name}')
    all_filenames['fnn'][name] = fnn_test.run(epochs['fnn'], optimizer, log=True, trials=trials['fnn'], tag=name)
for name, optimizer in cnn_optimizer.items():
    print(f'Train CNN using optimizer {name}')
    all_filenames['cnn'][name] = cnn_test.run(epochs['cnn'], optimizer, log=True, trials=trials['cnn'], tag=name)
'''
for name, optimizer in mf_optimizer.items():
    print(f'Train MF using optimizer {name}')
    all_filenames['mf'][name] = mf_test.run(epochs['mf'], optimizer, log=True, trials=trials['mf'], tag=name)

all_filenames['fnn']['gd']='./log/FFNN-gd-11-24T03:42:28'
all_filenames['fnn']['agd']='./log/FFNN-agd-11-24T04:27:18'
all_filenames['fnn']['agd_noise']='./log/FFNN-agd_noise-11-24T05:12:12'
all_filenames['fnn']['agd_NCE']='./log/FFNN-agd_NCE-11-24T05:57:02'
all_filenames['fnn']['all']='./log/FFNN-all-11-24T06:41:53'
all_filenames['cnn']['gd'] = './log/CNN-gd-11-24T07:00:39'
all_filenames['cnn']['agd'] = './log/CNN-agd-11-24T07:38:06'
all_filenames['cnn']['agd_noise'] = './log/CNN-agd_noise-11-24T08:15:14'
all_filenames['cnn']['agd_NCE'] = './log/CNN-agd_NCE-11-24T08:55:03'
all_filenames['cnn']['all'] = './log/CNN-all-11-24T09:34:29'

# save average data
for fname, filenames in all_filenames.items():
    print_('==================')
    print_(fname)
    print_(f'epochs = {epochs[fname]}')
    print_(f'trials = {trials[fname]}')
    print_('==================')
    for name, filename in filenames.items():
        logs = load_logs([filename])[0]
        print_(f'optimizer [{name}]')
        print_(f'{logs["params"]}')
        mean_train = np.mean(logs['train_losses'], axis=0)[epochs[fname]-1]
        std_train = np.std(logs['train_losses'], axis=0)[epochs[fname]-1]
        print_(f'mean_train_loss = {mean_train}')
        print_(f'std_train_loss = {std_train}')
        if not fname == 'cnn':
            mean_test = np.mean(logs['test_losses'], axis=0)[epochs[fname]-1]
            std_test = np.std(logs['test_losses'], axis=0)[epochs[fname]-1]
            print_(f'mean_test_loss = {mean_test}')
            print_(f'std_test_loss = {std_test}')

# make loss graph
for fname, filenames in all_filenames.items():
    logs = load_logs([filenames['gd'], filenames['agd_noise']])
    fig, ax = plt.subplots()
    ax.margins(0)
    ax.set_xlabel('epochs')
    ax.set_ylabel('losses')
    ax.plot(np.mean(logs[0]['train_losses'], axis=0), label='GD (Train)', color='#1f77b4')
    ax.plot(np.mean(logs[1]['train_losses'], axis=0), label='AGD with noise (Train)', color='#ff7f0e')
    if not fname == 'cnn':
        ax.plot(np.mean(logs[0]['test_losses'], axis=0), '--', label='GD (Test)', color='#1f77b4')
        ax.plot(np.mean(logs[1]['test_losses'], axis=0), '--', label='AGD with noise (Test)', color='#ff7f0e')
    else:
        ax.set_ylim(top=4) 
    ax.legend()
    fig.savefig(fname+"_agd_noise.png", dpi=600)

