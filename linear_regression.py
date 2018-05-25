import numpy as np
import torch
from torch import nn
from torch.autograd import Variable 

x_train = np.array(range(11), dtype=np.float32)
x_train = np.expand_dims(x_train, axis=1)
y_train = 2*x_train + 1

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim, output_dim)

if torch.cuda.is_available():
    print("Using GPU!\n")
    model.cuda()
else:
    print("Not using GPU.\n")

criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for epoch in range(epochs):
    epoch+=1
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(x_train).cuda())
        labels = Variable(torch.from_numpy(y_train).cuda())
    else:
        inputs = Variable(torch.from_numpy(x_train))
        labels = Variable(torch.from_numpy(y_train))
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.data[0]))

print('\nPredictions:')
predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
print(predicted)
print('\nGround truth:')
print(y_train)
