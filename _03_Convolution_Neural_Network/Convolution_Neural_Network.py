# 在该文件NeuralNetwork类中定义你的模型 
# 在自己电脑上训练好模型，保存参数，在这里读取模型参数（不要使用JIT读取），在main中返回读取了模型参数的模型

import os

os.system("sudo pip3 install torch")
os.system("sudo pip3 install torchvision")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.data import DataLoader
    

class NeuralNetwork(nn.Module):
    def init(self):
        super(NeuralNetwork, self).init()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)  # 调用预训练的ResNet18模型
    # 冻结ResNet中的参数，避免训练时梯度反向传递影响到预训练的模型
        for param in self.resnet18.parameters():
            param.requires_grad = False
    # 重新定义ResNet的输出层
        self.fc = nn.Linear(512, 10)


def forward(self, x):
    x = self.resnet(x)
    return x

def read_data():
    # 这里可自行修改数据预处理，batch大小也可自行调整
    # 保持本地训练的数据读取和这里一致
    dataset_train = torchvision.datasets.CIFAR10(root='../data/exp03', train=True, download=True, transform=torchvision.transforms.ToTensor())
    dataset_val = torchvision.datasets.CIFAR10(root='../data/exp03', train=False, download=False, transform=torchvision.transforms.ToTensor())
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=256, shuffle=False)
    return dataset_train, dataset_val, data_loader_train, data_loader_val

def train(model, data_loader_train, data_loader_val, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        
        for inputs, labels in data_loader_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_accuracy += torch.sum(preds == labels.data)
            
        train_loss = train_loss / len(data_loader_train.dataset)
        train_accuracy = train_accuracy / len(data_loader_train.dataset)
        
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        
        with torch.no_grad():
            for inputs, labels in data_loader_val:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_accuracy += torch.sum(preds == labels.data)
                
        val_loss = val_loss / len(data_loader_val.dataset)
        val_accuracy = val_accuracy / len(data_loader_val.dataset)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f} Val Accuracy: {val_accuracy:.4f}")
        print()
    
    # 保存模型参数
    torch.save(model.state_dict(), '../pth/model.pth')
    
def main():
    dataset_train, dataset_val, data_loader_train, data_loader_val = read_data()
    model = NeuralNetwork() # 若有参数则传入参数
    train(model, data_loader_train, data_loader_val, epochs=10, learning_rate=0.001)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model.load_state_dict(torch.load(parent_dir + '/pth/model.pth'))
    return model
    
