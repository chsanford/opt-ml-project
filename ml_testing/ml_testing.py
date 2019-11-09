# Followed this tutorial: https://nextjournal.com/gkoehler/pytorch-mnist
import time
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

    def run(self, n_epochs, optimizer, train_loader, test_loader, network, loss, sgd):
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

        self.test()
        print(f'Training for {n_epochs} epochs! are we doing sgd? {sgd}')
        start = time.time()

        for epoch in range(1, n_epochs + 1):
            self.train(epoch)
            self.test()

        print(f'Training took {time.time() - start} seconds.')
        fig = plt.figure()
        plt.plot(self.train_counter, self.train_losses, color='blue')
        plt.scatter(self.test_counter, self.test_losses, color='red')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        plt.xlabel('number of training examples seen')
        plt.ylabel('loss')
        plt.show()

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
                avg_loss = total_loss / (self.log_interval if self.sgd else n_train)

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t{}: {:.6f}'.format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    n_train,
                    100. * batch_idx / len(self.train_loader),
                    f'{self.loss_metric.__name__} {self.loss.__name__}',
                    self.loss_metric(avg_loss)))

                self.train_losses.append(self.loss_metric(avg_loss))
                self.train_counter.append(len(data) * batch_idx + (epoch - 1) * n_train)
                total_loss = 0
                # torch.save(self.network.state_dict(), 'results/mnist/model.pth')
                # torch.save(self.optimizer.state_dict(), 'results/mnist/optimizer.pth')

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
            print('\nTest set: {}: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                f'{self.loss_metric.__name__} {self.loss.__name__}',
                test_loss,
                correct,
                len(self.test_loader.dataset),
                100. * correct / len(self.test_loader.dataset)))
        else:
            print(f'\nTest set: {self.loss_metric.__name__} {self.loss.__name__}: {test_loss:.4f}\n')
