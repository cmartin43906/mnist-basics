import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# object to convert PIL image to tensor
transform = transforms.ToTensor()

# instance of MNIST dataset
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)

# train_data[i] returns tuple (tensor, label)
image, label = train_data[0]
print('Label:', label)

# .imshow expects 2D for greyscale
plt.imshow(image.squeeze(), cmap='gray')
plt.show()