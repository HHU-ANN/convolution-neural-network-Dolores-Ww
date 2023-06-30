# 在该文件NeuralNetwork类中定义你的模型 
# 在自己电脑上训练好模型，保存参数，在这里读取模型参数（不要使用JIT读取），在main中返回读取了模型参数的模型

import os

os.system("sudo pip3 install torch")
os.system("sudo pip3 install torchvision")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# 定义神经网络模型
class ResNet3(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet3, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.fc = nn.Linear(64*8*8, num_classes)
		
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

# 初始化模型和定义损失函数与优化器
model = ResNet3()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 加载CIFAR-10数据集
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor()
])
def read_data():
    train_dataset =  torchvision.datasets.CIFAR10(root='../data/exp03', train=True, download=True, transform=torchvision.transforms.ToTensor())
    train_loader =  DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
    test_dataset = torchvision.datasets.CIFAR10(root='../data/exp03', train=False, download=False, transform=torchvision.transforms.ToTensor())
    test_loader =  DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)
    return train_dataset,train_loader,test_dataset,test_loader


# 进行模型训练
train_dataset,train_loader,test_dataset,test_loader = read_data()
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), running_loss / 100))
            running_loss = 0.0
			





# 保存模型参数
torch.save(model.state_dict(), "pth/model.pth")

def main():
    model = NeuralNetwork()
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(parent_dir, 'pth')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    model.load_state_dict(torch.load(model_path))
    return model

if __name__ == '__main__':
    main()
