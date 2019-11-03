import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

from ml_testing.ml_testing import MLTest

view_example_images = False

# Defines network
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(28 * 28, 10)

	def forward(self, x):
		x = x.view(-1, 28 * 28)
		x = self.fc1(x)
		return F.log_softmax(x)

class MnistTest(MLTest):
	def __init__(self):
		self.network = Net()
		MLTest.__init__(self)

	def run(self, n_epochs, optimizer):
		if view_example_images:
			self.visualize_data()
		super().run(n_epochs, optimizer, self.get_train_loader(), self.get_test_loader(), self.network)

	# Loads training data
	def get_train_loader(self):
		batch_size_train = 60000 # all samples
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

	# Loads testing data
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

	# Visualizes sample data
	def visualize_data(self):
		examples = enumerate(test_loader)
		batch_idx, (example_data, example_targets) = next(examples)
		fig = plt.figure()
		for i in range(6):
			plt.subplot(2,3,i+1)
			plt.tight_layout()
			plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
			plt.title("Ground Truth: {}".format(example_targets[i]))
			plt.xticks([])
			plt.yticks([])
		plt.show()
