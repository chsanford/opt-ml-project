# Followed this tutorial: https://nextjournal.com/gkoehler/pytorch-mnist
# Custom descent modeled on:
# 	- https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
# 	- https://discuss.pytorch.org/t/simulated-annealing-custom-optimizer/38609
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required
import torchvision
import matplotlib.pyplot as plt

# Sets key constants
n_epochs = 10
batch_size_train = 60000 # all samples
batch_size_test = 1000
learning_rate = 0.1
log_interval = 10

view_example_images = False

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Loads training data
train_loader = torch.utils.data.DataLoader(
	torchvision.datasets.MNIST(
		'../data/mnist/train',
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

# Loads testing data
test_loader = torch.utils.data.DataLoader(
	torchvision.datasets.MNIST(
		'../data/mnist/test',
		train=False,
		download=True,
		transform=torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize((0.1307,), (0.3081,))
		])
	),
	batch_size=batch_size_test,
	shuffle=True)

# See some example images (if triggered)
if view_example_images:
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

# Defines network
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(28 * 28, 10)

	def forward(self, x):
		x = x.view(-1, 28 * 28)
		x = self.fc1(x)
		return F.log_softmax(x)

# Defines vanilla gradient descent optimizer
class GradientDescent(Optimizer):
	def __init__(self, params, learning_rate=0.1):
		defaults = dict(learning_rate=learning_rate)
		# super(GradientDescent, self).__init__(params, defaults)
		Optimizer.__init__(self, params, defaults)

	def step(self, closure=None):
		loss = None
		if closure is not None:
			loss = closure()
		
		for group in self.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue
				p.data.add_(-group['learning_rate'], p.grad.data)

		return loss

# Initialize model
network = Net()
optimizer = GradientDescent(
	network.parameters(),
	learning_rate=learning_rate
)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
	network.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		optimizer.zero_grad()
		output = network(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
		        epoch, batch_idx * len(data), len(train_loader.dataset),
		        100. * batch_idx / len(train_loader), loss.item()))
			train_losses.append(loss.item())
			train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
			torch.save(network.state_dict(), '../results/mnist/model.pth')
			torch.save(optimizer.state_dict(), '../results/mnist/optimizer.pth')

def test():
	network.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			output = network(data)
			test_loss += F.nll_loss(output, target, size_average=False).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(test_loader.dataset)
	test_losses.append(test_loss)
	print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
	    test_loss, correct, len(test_loader.dataset),
	    100. * correct / len(test_loader.dataset)))

test()
for epoch in range(1, n_epochs + 1):
	train(epoch)
	test()

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()