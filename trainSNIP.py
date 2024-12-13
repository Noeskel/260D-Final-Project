import torchvision.models as models
from torch import nn
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import CustomDataset
from snip_pruner import *
import time
import os
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        for img, label in dataloader:
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
        if e % 5 == 0:
            v_acc, v_loss = validate(model, val_dataloader, l_func)
        
        #data.subset(first)#get_subset(model, data, l_func))
        
        epoch_loss = running_loss / num_batches
        print(f'Epoch {e+1}: Loss = {epoch_loss:.7f}')
        if e % 10 == 0:
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

def apply_pruning(model, pruning_amount):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=pruning_amount)
    return model

def remove_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.remove(module, "weight")
    return model

def measure_sparsity(model):
    zeros = 0
    elements = 0
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            for param_name, param in module.named_parameters():
                if "weight" in param_name:
                    zeros += torch.sum(param == 0).item()
                    elements += param.nelement()
    sparsity = zeros / elements
    return zeros, elements, sparsity

if __name__ == '__main__':
    resnet = models.resnet18(pretrained=False)
    resnet.fc =  nn.Sequential(nn.Linear(512,400),
			nn.ReLU(inplace=True),
			nn.Linear(400,256),
                        nn.ReLU(inplace=True),
			nn.Linear(256, 10))
    #resnet = ResNet18()
    #resnet.load_state_dict(torch.load('./weights_b128_cifar/300_weights.pt', weights_only=True))
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean=mean, std=std)])
    train_data = datasets.CIFAR10('./cifar10', train=True, download=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False)
    test_data = datasets.CIFAR10('./cifar10', train=False, download=True, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    snip_pruner = Snip_Pruner(resnet, criterion, train_dataloader)
    pruned_model, masks = snip_pruner.prune(sparsity=0.8)
    torch.save(pruned_model.state_dict(), 'snip_model.pt')
    print(measure_sparsity(pruned_model))
    
    f = open('RuntimeSNIP.txt', 'a')
    #scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=3, threshold=0.001, mode='max')
    if not os.path.isdir('./SNIP80/Control/'):
        os.mkdir('./SNIP80/Control/')
    pruned_model.load_state_dict(torch.load('snip_model.pt'), strict=False)
    optimizer = torch.optim.Adam(pruned_model.parameters(), lr=0.01)
    start = time.time()
    f.write('Control'+'\n')
    f.write('Sparsity Before' + str(measure_sparsity(pruned_model))+'\n')
    #train(pruned_model, train_data, test_dataloader, criterion, optimizer, None, 50, './SNIP80/Control')
    end = time.time()
    f.write('Time (s):' + str(end-start)+'\n')
    f.write('Validity score' + str(validate(pruned_model, test_dataloader, criterion))+'\n')
    f.write('Sparsity' + str(measure_sparsity(pruned_model))+'\n'+'\n')
    f.close()
    subsetlist = ['Craig10']
    for i in range(len(subsetlist)):
        pruned_model.load_state_dict(torch.load('snip_model.pt'), strict=False)
        path = './SNIP80/'+subsetlist[i]
        subset = torch.load('./datasets/'+subsetlist[i]+'.pt', weights_only=False)
        if not os.path.isdir(path):
            os.mkdir(path)
        f = open('RuntimeSNIP.txt', 'a')
        start = time.time()
        f.write(subsetlist[i]+'\n')
        f.write('Sparsity Before' + str(measure_sparsity(pruned_model))+'\n')
        train_subset(pruned_model, subset, test_dataloader, criterion, optimizer, None, 50, path) #300,n
        end = time.time()
        f.write('Time (s):' + str(end-start)+'\n')
        f.write('Validity score' + str(validate(pruned_model, test_dataloader, criterion))+'\n')
        f.write('Sparsity' + str(measure_sparsity(pruned_model))+'\n'+'\n')
        f.close()
    
