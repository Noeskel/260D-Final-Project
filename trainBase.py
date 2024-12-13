import torchvision.models as models
from torch import nn
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import CustomDataset
import time
import os

def validate(model, dataloader, l_func, device='cuda'):
    model.eval()
    model = model.to(device)
    count = 0

    num_batches = 0
    running_loss = 0

    for img, label in dataloader:
        img, label = img.to(device), label.to(device)
        pred = model(img)
        loss = l_func(pred, label)
        running_loss += loss.item()
        pred = torch.argmax(pred, dim=1)
        count += torch.sum(pred == label).item()
        num_batches += 1

    print(f'Val Acc: {count / len(dataloader.dataset)}')
    return count / len(dataloader.dataset), running_loss / num_batches

def zscore(tensor):
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    return (tensor - mean) / std

def get_subset(model, data, l_func, device='cuda'):
    all_grads = torch.zeros((2, len(data.dataset)))
    dataloader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=False)
    for img, label, indexes in dataloader:
        img, label = img.to(device), label.to(device)
        pred = model(img)
        loss = l_func(pred, label)
        gradients = torch.sum(torch.abs(torch.autograd.grad(loss, model[-1].weight)))
        all_grads[0, indexes] = F.softmax(loss)*gradients
        all_grads[1, indexes] = label

    for i in range(10):
        z_i = zscore(all_grads[0, all_grads[1,:] == i])
        
    return indexes

def train(model, data, val_dataloader, l_func, optim, scheduler, epoch, save_path, device='cuda'):
    model = model.to(device)
    for e in range(epoch):
        model.train()
        num_batches = 0
        running_loss = 0
        dataloader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True)

        for img, label, indexes in dataloader:
            optim.zero_grad()
            img, label = img.to(device), label.to(device)
            pred = model(img)
            loss = l_func(pred, label)
            running_loss += loss.item()
            num_batches += 1
            if num_batches % 100 == 0:
	            print(f'Batch Loss: {loss.item()}')
            loss.backward()
            optim.step()
        v_acc = 0
        if e % 5 == 0 or e==9:
            v_acc, v_loss = validate(model, val_dataloader, l_func)
        
        #data.subset(first)#get_subset(model, data, l_func))
        
        epoch_loss = running_loss / num_batches
        print(f'Epoch {e+1}: Loss = {epoch_loss:.7f}')
        if e % 10 == 0 or e==9:
            torch.save(model.state_dict(), f'{save_path}/{e+1}_weights.pt')
        with open(f'{save_path}/loss.txt', 'a') as txt:
            txt.write(f'Epoch {e+1}: Loss = {epoch_loss:.7f} Val Acc = {v_acc}\n')
            
def train_subset(model, data, val_dataloader, l_func, optim, scheduler, epoch, save_path, device='cuda'):
    model = model.to(device)
    for e in range(epoch):
        model.train()
        num_batches = 0
        running_loss = 0
        dataloader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True)

        for img, label in dataloader:
            optim.zero_grad()
            img, label = img.to(device), label.to(device)
            pred = model(img)
            loss = l_func(pred, label)
            running_loss += loss.item()
            num_batches += 1
            if num_batches % 10 == 0:
	            print(f'Batch Loss: {loss.item()}')
            loss.backward()
            optim.step()
        v_acc = 0
        if e % 5 == 0:
            v_acc, v_loss = validate(model, val_dataloader, l_func)
        
        #data.subset(first)#get_subset(model, data, l_func))
        
        epoch_loss = running_loss / num_batches
        print(f'Epoch {e+1}: Loss = {epoch_loss:.7f}')
        if e % 10 == 0:
            torch.save(model.state_dict(), f'{save_path}/{e+1}_weights.pt')
        with open(f'{save_path}/loss.txt', 'a') as txt:
            txt.write(f'Epoch {e+1}: Loss = {epoch_loss:.7f} Val Acc = {v_acc}\n')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    resnet = models.resnet18(pretrained=False)
    resnet.fc =  nn.Sequential(nn.Linear(512,400),
			nn.ReLU(inplace=True),
			nn.Linear(400,256),
                        nn.ReLU(inplace=True),
			nn.Linear(256, 10))
    torch.save(resnet.state_dict(), 'originals.pth')
    #resnet = ResNet18()
    #resnet.load_state_dict(torch.load('./weights_b128_cifar/300_weights.pt', weights_only=True))
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean=mean, std=std)])
    train_data = CustomDataset()
    test_data = datasets.CIFAR10('./cifar10', train=False, download=True, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)
    optimizer = torch.optim.Adam(resnet.parameters(), lr=0.01)#torch.optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
    criterion = nn.CrossEntropyLoss()
    #scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=3, threshold=0.001, mode='max')
    if not os.path.isdir('./Baseline/Control/'):
        os.mkdir('./Baseline/Control/')
    f = open('Runtime.txt', 'a')
    start = time.time()
    f.write('Control'+'\n')
    train(resnet, train_data, test_dataloader, criterion, optimizer, None, 50, './Baseline/Control')
    end = time.time()
    f.write('Time (s):' + str(end-start)+'\n')
    f.write('Validity score' + str(validate(resnet, test_dataloader, criterion))+'\n')
    f.close()
    subsetlist = ['Craig50', 'Craig25', 'Craig10', 'Craig5', 'Craig1']
    for i in range(len(subsetlist)):
        resnet.load_state_dict(torch.load('originals.pth'))
        path = './Baseline/'+subsetlist[i]
        subset = torch.load('./datasets/'+subsetlist[i]+'.pt', weights_only=False)
        if not os.path.isdir(path):
            os.mkdir(path)
        f = open('Runtime.txt', 'a')
        start = time.time()
        f.write(subsetlist[i]+'\n')
        train_subset(resnet, subset, test_dataloader, criterion, optimizer, None, 50, path) #300,n
        end = time.time()
        f.write('Time (s):' + str(end-start)+'\n')
        f.write('Validity score' + str(validate(resnet, test_dataloader, criterion))+'\n')
        f.close()
    
