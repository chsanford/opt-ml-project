import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from ml_testing.ml_testing import MLTest
from models.ffnn import FFNN
from models.cnn import CNN


"""
Instance of MLTest that creates a Neural Network (FFNN or CNN) model.
Loads the MNIST dataset from torchvision for testing.
"""

class MnistTest(MLTest):
    view_example_images = False
    loss = F.nll_loss


    def __init__(self, ff=True):
        MLTest.__init__(self)
        self.model = FFNN() if ff else CNN()
        self.cnn = (not ff)
        self.dataset = torchvision.datasets.MNIST(
                'data/mnist/train',
                train=True,
                download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                    ])
                )
        print('Loading dataset...')
        self.data = next(iter(DataLoader(self.dataset, batch_size=len(self.dataset))))
        print('Finished loading the dataset into memory...') 


    def run(self, n_epochs, optimizer, sgd=False, log=False, trials=1, tag=''):
        if self.view_example_images:
            self.visualize_data()
        self.model.reset_parameters()
        return super().run(n_epochs,
                    self.model,
                    optimizer,
                    self.get_train_loader(sgd),
                    self.get_test_loader(),
                    MnistTest.loss,
                    sgd, trials=trials, tag=tag,
                    log=log)


    def get_train_loader(self, sgd=False):
        if not self.cnn:
            return self.data
        else:
            ret = []
            ret.append(self.data[0][range(3000)])
            ret.append(self.data[1][range(3000)])
            return ret
        '''
        batch_size_train = 1 if sgd else 60000  # all samples
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                'data/mnist/train',
                train=True,
                download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                ])
            ),
            batch_size=batch_size_train,
            shuffle=True
        )
        return train_loader
        '''

    def get_test_loader(self):
        batch_size_test = 1000
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                'data/mnist/test',
                train=False,
                download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                ])
            ),
            batch_size=batch_size_test,
            shuffle=True)
        return test_loader


    # Visualizes a sample of the test data.
    def visualize_data(self):
        examples = enumerate(test_loader)
        batch_idx, (example_data, example_targets) = next(examples)
        fig = plt.figure()
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
            plt.title("Ground Truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])
        plt.show()
