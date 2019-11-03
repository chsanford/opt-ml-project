# Followed this tutorial: https://nextjournal.com/gkoehler/pytorch-mnist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

log_interval = 10

class MLTest():

	def __init__(self):
		random_seed = 1
		torch.backends.cudnn.enabled = False
		torch.manual_seed(random_seed)

	def run(self, n_epochs, optimizer, train_loader, test_loader, network):
		self.optimizer = optimizer
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.network = network

		self.train_losses = []
		self.train_counter = []
		self.test_losses = []
		self.test_counter = [i*len(self.train_loader.dataset) for i in range(n_epochs + 1)]

		self.test()
		for epoch in range(1, n_epochs + 1):
			self.train(epoch)
			self.test()

		fig = plt.figure()
		plt.plot(self.train_counter, self.train_losses, color='blue')
		plt.scatter(self.test_counter, self.test_losses, color='red')
		plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
		plt.xlabel('number of training examples seen')
		plt.ylabel('negative log likelihood loss')
		plt.show()

	def train(self, epoch):
		self.network.train()
		for batch_idx, (data, target) in enumerate(self.train_loader):
			self.optimizer.zero_grad()
			output = self.network(data)
			loss = F.nll_loss(output, target)
			loss.backward()
			self.optimizer.step()
			if batch_idx % log_interval == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			        epoch, batch_idx * len(data), len(self.train_loader.dataset),
			        100. * batch_idx / len(self.train_loader), loss.item()))
				self.train_losses.append(loss.item())
				self.train_counter.append((batch_idx*64) + ((epoch-1)*len(self.train_loader.dataset)))
				torch.save(self.network.state_dict(), 'results/mnist/model.pth')
				torch.save(self.optimizer.state_dict(), 'results/mnist/optimizer.pth')

	def test(self):
		self.network.eval()
		test_loss = 0
		correct = 0
		with torch.no_grad():
			for data, target in self.test_loader:
				output = self.network(data)
				test_loss += F.nll_loss(output, target, size_average=False).item()
				pred = output.data.max(1, keepdim=True)[1]
				correct += pred.eq(target.data.view_as(pred)).sum()
		test_loss /= len(self.test_loader.dataset)
		self.test_losses.append(test_loss)
		print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		    test_loss, correct, len(self.test_loader.dataset),
		    100. * correct / len(self.test_loader.dataset)))
