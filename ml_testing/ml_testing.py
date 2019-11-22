# Followed this tutorial: https://nextjournal.com/gkoehler/pytorch-mnist
import datetime
import time
import os
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt


class MLTest():
    log_interval = 10000

    def __init__(self):
        random_seed = 1
        torch.backends.cudnn.enabled = False
        torch.manual_seed(random_seed)

    def run(self, n_epochs, optimizer, train_loader, test_loader, network, loss, sgd, save_model=False, log=False):
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.network = network
        self.loss = loss
        self.sgd = sgd
        self.train_losses = []
        self.train_counter = []
        self.test_losses = []
        self.test_counter = [i * len(self.train_loader.dataset) for i in range(n_epochs + 1)]
        idt = lambda x: x
        idt.__name__ = ''
        self.loss_metric = math.sqrt if self.loss == F.mse_loss else idt  # report RMSE if doing matrix factorization

        print(f'Model: {type(self.network).__name__}\nParams:{self.optimizer.get_params()}\n')

        start = time.time()
        self.epoch_start = start
        self.test()
        print(f'Training for {n_epochs} epochs! sgd? {sgd} / logging? {log}')

        for epoch in range(1, n_epochs + 1):
            self.epoch_start = time.time()
            self.train(epoch)
            self.test()

        if save_model:
            self._save_model()

        if log:
            self._log_training()

        print(f'Training took {time.time() - start} seconds.')
        fig = plt.figure()
        plt.plot(self.train_counter, self.train_losses, color='blue')
        plt.scatter(self.test_counter, self.test_losses, color='red')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        plt.xlabel('number of training examples seen')
        plt.ylabel('loss')
        plt.show()


    def _log_training(self):
        fname = f'./log/train-{datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}'

        if not os.path.exists('./log'):
            os.makedirs('./log')

        with open(fname, 'wb') as logfile:
            print(f'Saving training log to {fname}')
            d = {'name': type(self.network).__name__,
                 'loss': f'{self.loss_metric.__name__} {self.loss.__name__}',
                 'params': self.optimizer.get_params(),
                 'train_losses': self.train_losses,
                 'test_losses': self.test_losses}
            pickle.dump(d, logfile)


    def _save_model(self):
        raise NotImplementedError()  # implement in subclass


    def train(self, epoch):
        total_loss = 0
        n_train = len(self.train_loader.dataset)
        self.network.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            def closure():
                self.optimizer.zero_grad()
                output = self.network(data)
                loss = self.loss(output, target)
                loss.backward()
                return loss

            loss = self.optimizer.step(closure)
            total_loss += loss.item()

            if not self.sgd or (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / (self.log_interval if self.sgd else 1)

                print('Epoch: {} [{}/{} ({:.0f}%)]\t{}: {:.4f}'.format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    n_train,
                    100. * batch_idx / len(self.train_loader) if self.sgd else 100.,
                    f'{self.loss_metric.__name__} {self.loss.__name__}',
                    self.loss_metric(avg_loss)))

                self.train_losses.append(self.loss_metric(avg_loss))
                self.train_counter.append(len(data) * batch_idx + (epoch - 1) * n_train)
                total_loss = 0


    def test(self):
        self.network.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.network(data)
                test_loss += self.loss(output, target, reduction='sum').item()
                if self.loss == F.nll_loss:
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()

        test_loss = self.loss_metric(test_loss / len(self.test_loader.dataset))
        self.test_losses.append(test_loss)

        if self.loss == F.nll_loss:
            print('Test set: {}: {:.4f}, Accuracy: {}/{} ({:.0f}%), dur: {:.2f}'.format(
                f'{self.loss_metric.__name__} {self.loss.__name__}',
                test_loss,
                correct,
                len(self.test_loader.dataset),
                100. * correct / len(self.test_loader.dataset),
                time.time() - self.epoch_start))
        else:
            print(f'Test set: {self.loss_metric.__name__} {self.loss.__name__}: {test_loss:.4f}, dur: {time.time() - self.epoch_start:.2f}\n')
