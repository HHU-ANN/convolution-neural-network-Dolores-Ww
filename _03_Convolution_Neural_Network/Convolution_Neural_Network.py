# 在该文件NeuralNetwork类中定义你的模型 
# 在自己电脑上训练好模型，保存参数，在这里读取模型参数（不要使用JIT读取），在main中返回读取了模型参数的模型

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Define your neural network architecture here

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(4 * 4 * 256, 10) # Output layer

    def forward(self, x):
        # Implement the forward pass of your neural network here

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.view(-1, 4 * 4 * 256) # Flatten the input tensor
        x = self.fc(x)

        return x



def read_data():
    dataset_train = torchvision.datasets.CIFAR10(root='./data/exp03', train=True, download=True, transform=torchvision.transforms.ToTensor())
    dataset_val = torchvision.datasets.CIFAR10(root='./data/exp03', train=False, download=False, transform=torchvision.transforms.ToTensor())
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=1, shuffle=False)
    return dataset_train, dataset_val, data_loader_train, data_loader_val


def main():
    model = NeuralNetwork()
    dataset_train, dataset_val, data_loader_train, data_loader_val = read_data()
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(parent_dir, 'pth')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    model.load_state_dict(torch.load(model_path))
    return model

if __name__ == '__main__':
    main()

