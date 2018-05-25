import numpy as np
import torch
from torch import nn
from torchvision import transforms
import torchvision.datasets as dsets
from torch.autograd import Variable 

train_dataset = dsets.MNIST(root='./mnist_data',
        train=True,
        transform=transforms.ToTensor(),
        download=True)

test_dataset = dsets.MNIST(root='./mnist_data',
        train=False,
        transform=transforms.ToTensor(),
        download=True)

batch_size = 100
n_iters = 3000
num_epochs = int(n_iters / (len(train_dataset) / batch_size))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False)

class CNNModel(nn.Module):
    def __init__(self, input_size, hidden_dim, num_classes):
        super(CNNModel, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d()
        self.relu1= nn.ReLU()

        # TODO: Remaining layers


