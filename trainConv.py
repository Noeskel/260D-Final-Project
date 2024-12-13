import torchvision.models as models
from torch import nn
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import CustomDataset
import csv, os
import time
from prune import prune_model

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

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

def get_subset(loss, labels, indexes, class_loss, class_count, class_std):
    z = zscore(loss, (class_loss/torch.clamp(class_count, min=1e-5))[labels], class_std[labels])
    top_k = int(len(indexes)*.95)
    val, index = torch.topk(z, top_k)
    return indexes[index].flatten().tolist()

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

def group_stats(class_loss, class_count, class_std, loss, label):
    group_loss = torch.bincount(label, weights=loss, minlength=10)
    group_count = torch.bincount(label, minlength=10)
    group_std = torch.zeros(10, dtype=torch.float).cuda()
    for i in range(10):
        group_std[i] = torch.std(loss[label == i])

    t_mean = (class_loss + group_loss)/(class_count + group_count)
    class_std = (class_count-1)*class_std**2 + class_count*(class_loss/torch.clamp(class_count,min=1e-5)-t_mean)**2 + (group_count-1)*group_std**2 + group_count*(group_loss/torch.clamp(group_count,min=1e-5)-t_mean)**2
    
    class_count += group_count
    class_loss += group_loss
    class_std /= torch.clamp(class_count-1, min=1e-5)
    class_std = torch.sqrt(class_std)
    return class_count, class_loss, class_std


def train(model, data, l_func, optim, epoch, save_path='reg_10_20_prune', device='cuda'):
    #class_loss = torch.zeros(10, dtype=torch.float).cuda()
    #class_count = torch.zeros(10, dtype=torch.float).cuda()
    #class_std = torch.zeros(10, dtype=torch.float).cuda()
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model = model.to(device)
    
    model.train()
    for e in range(epoch):
        
        num_batches = 0
        running_loss = 0
        data_size = len(data)
        
        print('LEN DATA', data_size)
        dataloader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True)
        for img, label, indexes in dataloader:
            optim.zero_grad()
            img, label = img.to(device), label.to(device)
            pred = model(img)
            loss = l_func(pred, label) #+ 20 * regularizer(model) 

            running_loss += loss.item()
            num_batches += 1
            if num_batches % 10 == 0:
                print(f'Batch Loss: {loss.item()}')
            loss.backward()
            optim.step()
        
        epoch_loss = running_loss / num_batches
        print(f'Epoch {e+1}: Loss = {epoch_loss:.7f}')
        torch.save(model.state_dict(), f'{save_path}/{e+1}_weights.pt')
        with open(f'{save_path}/train_loss.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([e+1, epoch_loss, data_size])

def print_param(model):
    for name, param in model.named_modules():
        print(param)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == '__main__':
    resnet = models.resnet18(pretrained=False)
    resnet.fc =  nn.Sequential(nn.Linear(512,400),
			nn.ReLU(inplace=True),
			nn.Linear(400,256),
            nn.ReLU(inplace=True),
			nn.Linear(256, 10))
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    criterion = nn.CrossEntropyLoss()
    resnet.load_state_dict(torch.load('./10_15_weights.pt', weights_only=True))
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean=mean, std=std)])
    train_data = CustomDataset()
    test_data = datasets.CIFAR10('./cifar10', train=False, download=True, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)
    #resnet.load_state_dict(torch.load('./reg_15/10_weights.pt', weights_only=True))
    f=open('ModelPrunedResults.txt', 'a')
    f.write('Accuracy Before: '+str(validate(resnet, test_dataloader, criterion)[0])+'\n')
    before = count_parameters(resnet)
    print(before)
    resnet = prune_model(resnet, 6.02, 0)
    after = count_parameters(resnet)
    print(after)
    print("Reduction:", str((1-after/before)*100))
    f.write("Reduction: "+str((1-after/before)*100))
    f.write('Accuracy After Pruning: '+str(validate(resnet, test_dataloader, criterion)[0])+'\n')
    optimizer = torch.optim.Adam(resnet.parameters(), lr=0.01)

    #print_param(resnet)
    start = time.time()
    train(resnet, train_data, criterion, optimizer, 40)
    end = time.time()
    f.write('Training Time (seconds): '+str(end - start)+'\n')
    f.write('Accuracy After Prune+Train: '+str(validate(resnet, test_dataloader, criterion)[0])+'\n')
    f.close()
