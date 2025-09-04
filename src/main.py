# main lib for neural networks
import torch

# submod for nn, layer types, activation functions, base classes
import torch.nn as nn

# functions for activation or loss
import torch.nn.functional as F

# optimizers to adjust weights
import torch.optim as optim

# helper class to feed data in batches
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# DC inheirits from nn.Module
class DigitClassifier(nn.Module):
    # constructor
    def __init__(self):
        super().__init__()
        # create fully connected layers
        # flattening = turning 2d into 1d while keeping pixel values
        # nn.Linear = a matrix multiplication layer with a bias
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    # connect the layers with actual data flow
    # x is batch of images
    def forward(self, x):
        # .view is reshaping
        # infer batch size, flatten image into 1d vector (784)
        # fits expected input of fc1 (batch size, input features per sample)
        x = x.view(-1, 28*28)

        # fc1 and 2 are hidden layers
        # x = F.relu((batch size, 128 outputs)) (tensor)
        # ReLU(x) = max(0,x); activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # output layer produces raw scores (logits) for each class
        x = self.fc3(x)
        return x



# object to convert PIL image to tensor
transform = transforms.ToTensor()

# instance of MNIST dataset
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# batching data
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

# instance of nn
model = DigitClassifier()
# define loss function
criterion = nn.CrossEntropyLoss()
# define optimizer, stochastic gradient descent
# .parameters() collects tensors of layer weights
# 0.01 is size of adjustment applied to each weight on every update
optimizer = optim.SGD(model.parameters(), lr=0.01)

# grab a batch
images, labels = next(iter(train_loader))

#### training ####
num_epochs = 7

for epoch in range(num_epochs):

    # set to training mode      
    model.train()
    for images, labels in train_loader:
        # zero out gradients
        optimizer.zero_grad()

        # calls forward(images)
        outputs = model(images)

        # compute loss, tensor
        loss = criterion(outputs, labels)

        # backward pass, calculate gradients
        loss.backward()
        # changes the weights using gradients
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    #### testing ####

    # set to eval mode  
    model.eval()
    # tells pytorch not to track gradients
    with torch.no_grad():

        # counters
        total = 0
        correct = 0

        for images, labels in test_loader:
            # forward pass
            outputs = model(images)

            # ignore confidence values, capture predicted classification
            _, predicted = torch.max(outputs, 1)

            # creates tensor of T/F for each sample, counts True, converts to   scalar
            correct += (predicted == labels).sum().item()

            # returns size along a dimension of tensor
            total += labels.size(0)

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy*100:.2f}%")
# train_data[i] returns tuple (tensor, label)
# image, label = train_data[0]
# print('Label:', label)

# .imshow expects 2D for greyscale
# plt.imshow(image.squeeze(), cmap='gray')
# plt.show()