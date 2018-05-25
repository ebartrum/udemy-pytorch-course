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
n_iters = 5000
num_epochs = int(n_iters / (len(train_dataset) / batch_size))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False)

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_size, hidden_dim, num_classes):
        # Create the various layers as member vars
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linearity
        self.relu1= nn.ReLU()
        # Linear function
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2= nn.ReLU()
        # Linear function (readout)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Define the forward pass
        # Linear function
        out = self.fc1(x)
        # Non-linearity
        out = self.relu1(out)
        # Linear function (readout)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

input_dim = 28*28 
hidden_dim = 200
output_dim = 10

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()
learning_rate = 0.005
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print("Starting training...")
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.view(-1, 28*28).cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        iter += 1

        if iter % 50 == 0:
            correct = 0
            total = 0

            for images, labels in test_loader:
                if torch.cuda.is_available():
                    images = Variable(images.view(-1, 28*28).cuda())
                else:
                    images = Variable(images.view(-1, 28*28))
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()
            accuracy = correct / total
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, 
                loss.data[0], accuracy))
