import torchvision.models as models
from torch import nn
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import CustomDataset
import csv, os
import statistics, itertools
import time
import prune

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

def zscore(batch_grad, mean, std):
    return (batch_grad - mean) / std

def get_subset2(model, data, device='cuda'):
    keep_indexes = []
    all_grads = torch.zeros(len(data), dtype=torch.float)
    all_labels = torch.zeros(len(data), dtype=torch.long)
    dataloader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=False)
    l_func = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    for img, label, indexes in dataloader:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            loss = l_func(pred, label)
            gradient = 0
            for name, param in model.fc.named_parameters():
                if name == '4.weight':
                    gradient = torch.sum(torch.abs(torch.autograd.grad(torch.sum(loss), param)[0]))

            batch_grad = F.softmax(loss.detach(), dtype=torch.float)*gradient.detach()
            all_grads[indexes] = batch_grad.cpu()
            all_labels[indexes] = label.cpu()
    print('DONEEEEE')
    for i in range(10):
        grad_i = all_grads[all_labels == i]
        mean_i = torch.mean(grad_i)
        std_i = torch.std(grad_i)
        z = zscore(grad_i, mean_i, std_i)
        keep_indexes += torch.where(all_labels == i)[0][((z >= -3) & (z <= -0.1) | (z <= 3) & (z >= 0.1)).nonzero()].flatten().tolist() #(z >= -3) & (z <= -0.1) | (z <= 3) & (z >= 0.1) (z > -2.75) & (z < 2.75)
    
    print(len(keep_indexes))
    return torch.tensor(keep_indexes)

def get_subset(model, data, l_func, device='cuda'):
    keep_indexes = []
    print(f'NUM DATA: {len(data)}')
    running_avg = torch.zeros(10, dtype=torch.float).to(device)
    running_std = torch.zeros(10, dtype=torch.float).to(device)
    running_count = torch.zeros(10, dtype=torch.float).to(device)
    dataloader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=False)

    l_func = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    
    for img, label, indexes in dataloader:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            loss = l_func(pred, label)
            gradient = 0
            for name, param in model.fc.named_parameters():
                if name == '4.weight':
                    gradient = torch.sum(torch.abs(torch.autograd.grad(torch.sum(loss), param)[0]))

            
            prev_count = running_count
            prev_avg = running_avg
            batch_grad = F.softmax(loss.detach(), dtype=torch.float)*gradient.detach()

            batch_count = torch.bincount(label, minlength=10).detach()
            batch_avg = torch.bincount(label, weights=batch_grad, minlength=10).detach()/batch_count
            batch_var = torch.zeros(10, dtype=torch.float).to(device)

            running_count += batch_count
            running_avg = (prev_count*prev_avg + batch_count*batch_avg)/running_count

            for i in range(10):
                batch_i = batch_grad[label==i].tolist()
                if len(batch_i) > 1:
                    var_i = statistics.variance(batch_i)
                    batch_var[i] = var_i

            running_std = (prev_count-1)*running_std + (batch_count-1)*batch_var + prev_count*(prev_avg-running_avg)**2 + batch_count*(batch_avg-running_avg)**2
            running_std /= (running_count-1)

    running_std = torch.sqrt(running_std)
    print('DONEEEEE')
    first = True
    for img, label, indexes in dataloader:
        
            img, label = img.to(device), label.to(device)
            pred = model(img)
            loss = l_func(pred, label)

            gradient = 0
            for name, param in model.fc.named_parameters():
                if name == '4.weight':
                    gradient = torch.sum(torch.abs(torch.autograd.grad(torch.sum(loss), param)[0]))
                    if first:
                        print(gradient)
                        first = False
            
            batch_grad = F.softmax(loss.detach(), dtype=torch.float)*gradient.detach()
            z = zscore(batch_grad, running_avg[label], running_std[label]).cpu()
            #print(z.shape)
            #print(z)
            keep_indexes += indexes[((z >= -3) & (z <= 3)).nonzero()].tolist()
    
    print(len(keep_indexes))
    return keep_indexes



def sim_conv(x):
    out_channel = x.shape[0]
    score = 0
    for i in range(out_channel-1):
        score_ij = torch.mean(torch.abs(x[i, :] - x[i+1, :]))
        score += score_ij
    return score/(out_channel-1)

def sim_linear(x):
    return torch.mean(torch.abs(x - torch.mean(x, dim=0)))

def regularizer(model):
    linear_penalty = 0
    conv_penalty = 0
    linear_count = conv_count = 0
    for name, param in model.named_parameters():
        if 'conv' in name:
            conv_penalty += sim_conv(param)
            conv_count += 1
        if 'fc' in name and 'weight' in name:
            linear_penalty += sim_linear(param)
            linear_count += 1
    return linear_penalty/linear_count + conv_penalty/conv_count

def regularizer_l1(model):
    l1_penalty = 0
    for name, param in model.named_parameters():
        if 'conv' in name or ('fc' in name and 'weight' in name):
            l1_penalty += torch.norm(param, 1)
    return l1_penalty

def train(model, data, val_dataloader, l_func, optim, scheduler, epoch, save_path='baseline_custom_reg', device='cuda'):
    model.train()
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataloader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True)
    model = model.to(device)
    original_data_size = len(data)
    for e in range(epoch):
        num_batches = 0
        running_loss = 0
        data_size = len(data)
        #print('LEN DATA', data_size)
        
        #dataloader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True)
        for img, label, indexes in dataloader:
            optim.zero_grad()
            img, label = img.to(device), label.to(device)
            pred = model(img)
            loss = l_func(pred, label)
            if e < 10:
                loss = l_func(pred, label) + regularizer(model) 
            running_loss += loss.item()
            num_batches += 1
            print(f'Batch Loss: {loss.item()}')
            loss.backward()
            optim.step()
        v_acc = 0
        #if e % 5 == 0:
        #    v_acc, v_loss = validate(model, val_dataloader, l_func)
        #if (e+1) % 5 == 0 and data_size/original_data_size > 0.2:
        #    data.subset(get_subset2(model, data))
        
        epoch_loss = running_loss / num_batches
        print(f'Epoch {e+1}: Loss = {epoch_loss:.7f}')
        torch.save(model.state_dict(), f'{save_path}/{e+1}_weights.pt')
        with open(f'{save_path}/train_loss.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([e+1, epoch_loss, data_size])

def print_param(model):
    for name, param in model.named_modules():
        print(name)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

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
    resnet.load_state_dict(torch.load('10_weights.pt'), strict=False)
    pruned = prune.prune_model(resnet, 1)
    print(measure_sparsity(pruned_model))
