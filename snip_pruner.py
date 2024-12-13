import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import copy

def forward_conv2d(self, input):
    return F.conv2d(input, self.weight * self.weight_mask, self.bias, self.stride, self.padding, self.dilation, self.groups)

def forward_conv1d(self, input):
    return F.conv1d(input, self.weight * self.weight_mask, self.bias)

def forward_linear(self, input):
    return F.linear(input, self.weight * self.weight_mask, self.bias)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Snip_Pruner:
    def __init__(self, model, criterion, dataloader):
        self.model = copy.deepcopy(model).to(device)
        self.pruned_model = copy.deepcopy(model).to(device)
        self.criterion = criterion.to(device)
        self.dataloader = dataloader
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight).to(device))
                nn.init.xavier_normal_(layer.weight)
                layer.weight.requires_grad = False
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(forward_conv2d, layer)
            if isinstance(layer, nn.Conv1d):
                layer.forward = types.MethodType(forward_conv2d, layer)
            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(forward_linear, layer)
        
    def prune(self, sparsity):
        #single batch
        data, labels= next(iter(self.dataloader))
        self.model.zero_grad()
        data, labels = data.to(device), labels.to(device)
        output = self.model(data)
        loss = self.criterion(output, labels)
        loss.backward()
        grad_list = []
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
                grad_list.append(torch.abs(layer.weight_mask.grad))
        grads = torch.cat([torch.flatten(x) for x in grad_list])
        gradsum = torch.sum(grads)
        grads.div_(gradsum)
        topkparams = int((1-sparsity) * len(grads)) #Keep the top 1-sparsity params
        values, idxs = torch.topk(grads, topkparams, sorted=True) #Returns top tensor values and indices
        threshold = values[-1]
        masks = [(grad / gradsum > threshold).float() for grad in grad_list]
        
        prun_layers = filter(lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear), self.pruned_model.modules())
        
        def apply_masking(mask):
            def hook(weight):
                return weight * mask
            return hook
        
        for layer, mask in zip(prun_layers, masks):
            if layer.weight.shape == mask.shape:
                layer.weight.data = layer.weight.data * mask
                layer.weight.register_hook(apply_masking(mask))
            else:
                print('error')
                return
        return self.pruned_model, masks
    
