from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch

class CustomDataset(Dataset):
    def __init__(self):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean=mean, std=std)])
        self.set = datasets.CIFAR10(root='./cifar10', download=True, train=True, transform=transform)
        
    def __getitem__(self, index):
        data, target = self.set[index]

        return data, target, index

    def __len__(self):
        return len(self.set)
    
    def subset(self, indexes):
        self.set = torch.utils.data.Subset(self.set, indexes)