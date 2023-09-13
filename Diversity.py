import torch  
import torch.nn as nn  
import torch.optim as optim  
import torchvision  
import torchvision.transforms as transforms  
from torch.utils.data import DataLoader,Dataset
import umap 
import os
from PIL import Image
import matplotlib.pyplot as plt  

class dir_dataset(Dataset):
    def __init__(self,dir,transform=None) -> None:
        super(dir_dataset,self).__init__()
        self.dir=dir
        self.transform=transform

    def __len__(self):
        return len(self.name)
    
    def __getitem__(self, index):
        path=self.name[index]
        image = Image.open(os.path.join(self.dir,path))
        if self.transform:
            image=self.transform(image)
        
        return image,1


        
  
# 设置随机种子以确保结果可重现  
torch.manual_seed(42)  
  
# 检查是否可使用CUDA加速  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
  
# 定义数据预处理的转换  
transform = transforms.Compose([  
    transforms.RandomCrop(32, padding=4),  
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])  
  
# 加载CIFAR-10训练集和测试集  
# trainset = torchvision.datasets.CIFAR10(root='./data/cifar-10', train=True, download=True, transform=transform)  
# testset = torchvision.datasets.CIFAR10(root='./data/cifar-10', train=False, download=True, transform=transform)  
trainset=dir_dataset("/ibex/user/lij0w/codes/stablediffusion/results/layer8_trade0.1/computer programmer/origin",transform)
testset=dir_dataset("/ibex/user/lij0w/codes/stablediffusion/results/layer8_trade0.1/computer programmer/fair",transform)
  
# 定义数据加载器  
batch_size = 32 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)  
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)  
  
# 加载预训练的ResNet-18模型  
model = torchvision.models.resnet18(pretrained=True)  
num_classes = 10  
model.fc = nn.Linear(model.fc.in_features, num_classes)  
model = model.to(device)  
  
# 提取测试集特征并进行2维特征可视化  
def visualize_features(data, labels, title):  
    model.eval()  
    with torch.no_grad():  
        features = []  
        for images, _ in data:  
            images = images.to(device)  
            outputs = model(images)  
            features.extend(outputs.cpu().numpy())  
      
    reducer = umap.UMAP(n_components=2)  
    embedding = reducer.fit_transform(features)  
      
    plt.figure(figsize=(8, 8))  
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10')  
    plt.title(title)  
    plt.colorbar(ticks=range(10), label='Class')  
    plt.show()  
  
# 初始测试集特征可视化（3维）  
# visualize_features(testloader, testset.targets, "Initial CIFAR-10 Test Set Feature Visualization (2D)")  
  
# 定义损失函数和优化器  
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)  
  
# 训练微调模型  
num_epochs = 50  
  
for epoch in range(num_epochs):  
    running_loss = 0.0  
    total_accuracy = 0.0  
  
    for i, (inputs, labels) in enumerate(trainloader):  
        inputs = inputs.to(device)  
        labels = labels.to(device)  
  
        optimizer.zero_grad()  
  
        outputs = model(inputs)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()  
  
        running_loss += loss.item()  
  
        # 计算精度  
        _, predicted = torch.max(outputs, 1)  
        correct = (predicted == labels).sum().item()  
        total = labels.size(0)  
        accuracy = correct / total  
        total_accuracy += accuracy  
  
    epoch_loss = running_loss / len(trainloader)  
    epoch_accuracy = total_accuracy / len(trainloader)  
    print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f}")  
  
# 重新提取测试集特征并进行2维特征可视化  
visualize_features(testloader, testset.targets, "CIFAR-10 Test Set Feature Visualization (2D)")  