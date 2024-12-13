import torchvision.models as models
from torch import nn
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import CustomDataset
import torch.nn.utils.prune as prune
import time
import os
import copy

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
        if e % 5 == 0:
            v_acc, v_loss = validate(model, val_dataloader, l_func)
        
        #data.subset(first)#get_subset(model, data, l_func))
        
        epoch_loss = running_loss / num_batches
        print(f'Epoch {e+1}: Loss = {epoch_loss:.7f}')
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
        torch.save(model.state_dict(), f'{save_path}/{e+1}_weights.pt')
        with open(f'{save_path}/loss.txt', 'a') as txt:
            txt.write(f'Epoch {e+1}: Loss = {epoch_loss:.7f} Val Acc = {v_acc}\n')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def apply_pruning(model, pruning_amount, device = 'cuda'):
    pruned_model = copy.deepcopy(model).to(device)
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=pruning_amount)
    return pruned_model

def remove_pruning(model):
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.remove(module, "weight")
    return pruned_model

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
    train_data = CustomDataset()
    test_data = datasets.CIFAR10('./cifar10', train=False, download=True, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet.parameters(), lr=0.01)
    if not os.path.isdir('./OneShot/Control'):
        os.mkdir('./OneShot/Control')
    start10 = time.time()
    train(resnet, train_data, test_dataloader, criterion, optimizer, None, 5, './OneShot/Control')
    end10 = time.time()
    f = open('RuntimeOneShot5-45.txt', 'a')
    f.write('Time (s) for 5 epochs:' + str(end10-start10)+'\n')
    f.write('Validity score at 5 epochs before prune ' + str(validate(resnet, test_dataloader, criterion))+'\n')
    f.write('Sparsity' + str(measure_sparsity(resnet))+'\n'+'\n')
    torch.save(resnet.state_dict(), '5_weights.pt')
#    resnet.load_state_dict(torch.load('5_weights.pt'))
    startprune = time.time()
    pruned_model = apply_pruning(resnet, pruning_amount=0.2)
    pruned_model = remove_pruning(pruned_model)
    endprune = time.time()
    f.write('Time to prune:' + str(endprune-startprune)+'\n')
    torch.save(pruned_model.state_dict(), 'pruned_oneshot.pt')
    resnet.load_state_dict(torch.load('pruned_oneshot.pt'), strict=False)
    optimizer = torch.optim.Adam(pruned_model.parameters(), lr=0.01)#torch.optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
    f.write('Validity score after prune, before training' + str(validate(resnet, test_dataloader, criterion))+'\n')
    f.write('Sparsity' + str(measure_sparsity(resnet))+'\n'+'\n')
    f.write('Control'+'\n')
    start = time.time()
    train(resnet, train_data, test_dataloader, criterion, optimizer, None, 45, './OneShot/Control')
    end = time.time()
    f.write('Time (s) for rremaining 45 epochs:' + str(end-start)+'\n')
    f.write('Validity score' + str(validate(resnet, test_dataloader, criterion))+'\n')
    f.write('Sparsity' + str(measure_sparsity(resnet))+'\n'+'\n')
    f.close()
    subsetlist = ['Craig50', 'Craig25', 'Craig10', 'Craig5', 'Craig1']
    for i in range(len(subsetlist)):
        resnet.load_state_dict(torch.load('pruned_oneshot.pt'), strict=False)
        path = './OneShot/'+subsetlist[i]
        subset = torch.load('./datasets/'+subsetlist[i]+'.pt', weights_only=False)
        if not os.path.isdir(path):
            os.mkdir(path)
        f = open('RuntimeOneShot5-45.txt', 'a')
        start = time.time()
        f.write(subsetlist[i]+'\n')
        train_subset(resnet, subset, test_dataloader, criterion, optimizer, None, 45, path) #300,n
        end = time.time()
        f.write('Time (s):' + str(end-start)+'\n')
        f.write('Validity score' + str(validate(resnet, test_dataloader, criterion))+'\n')
        f.write('Sparsity' + str(measure_sparsity(resnet))+'\n'+'\n')
        f.close()    
