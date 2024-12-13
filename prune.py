import torch
from torch import nn

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        conv1,
        bn1,
        conv2,
        bn2,
        downsample=None,
    ):
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv2
        self.bn2 = bn2
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


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

def update_linear_structure(linear_layer, prev_keep_index=None, thresh=1):
    linear_weights = linear_layer.weights
    linear_bias = linear_layer.bias

    if prev_keep_index is not None:
        linear_weights = linear_weights[prev_keep_index, :]
        linear_bias = linear_bias[prev_keep_index]

    keep_index = torch.where(sim_linear(linear_weights) > thresh)
    prune_weights = linear_weights[:, keep_index]

    new_out_features, new_in_features = prune_weights.shape

    new_linear_layer = nn.Linear(new_in_features, new_out_features)
    new_linear_layer.weight.data = prune_weights
    new_linear_layer.bias.data = linear_bias

    return new_linear_layer, keep_index

def merge_conv(x, thresh):
    out_channel = x.shape[0]
    merge = []

    for i in range(out_channel-1):
            score_ij = torch.mean(torch.abs(x[i, :] - x[i+1, :]))
            if score_ij < thresh:
                merge.append([i,i+1])

    return merge

def update_conv_structure(conv_layer, prev_merge_pairs=None, thresh=1, residual=False):
    conv_weights = conv_layer.weight
    out_channels, in_channels, kernel_size, _ = conv_weights.shape

    if prev_merge_pairs is not None:
        prev_keep_index = range(in_channels)
        for merge_index, remove_index in prev_merge_pairs:
            conv_weights[:, merge_index, :, :] += conv_weights[:, remove_index, :, :]
            prev_keep_index.remove(remove_index)

        conv_weights = conv_weights[:, prev_keep_index, :, :]
    
    if residual:
        prune_weights = conv_weights
    else:
        merge_pairs = merge_conv(conv_weights, thresh)
    
        keep_index = range(out_channels)
        for _,j in merge_pairs:
            keep_index.remove(j)

        prune_weights = conv_weights[keep_index, :]
    new_out_channels, new_in_channels, _, _ = prune_weights.shape
    new_conv_layer = nn.Conv2d(new_in_channels, new_out_channels, kernel_size, stride=conv_layer.stride, padding=conv_layer.padding, bias=False, dilation=conv_layer.dilation)
    new_conv_layer.weight.data = prune_weights

    return new_conv_layer, merge_pairs

def update_residual_connection_structure(downsample_block, block_last_merge_pairs):
    new_conv_layer, _ = update_conv_structure(downsample_block[0], block_last_merge_pairs, residual=True)
    new_bn_layer, _ = update_bn_structure(downsample_block[1], block_last_merge_pairs)
    return new_conv_layer, new_bn_layer


def update_bn_structure(bn_layer,prev_merge_pairs=None):
    bn_weights = bn_layer.weight
    bn_bias = bn_layer.bias

    channels = bn_weights.shape[0]
    prev_keep_index = range(channels)
    if prev_merge_pairs is not None:
        for merge_index, remove_index in prev_merge_pairs:
            prev_keep_index.remove(remove_index)
        bn_weights = bn_weights[prev_keep_index]
        bn_bias = bn_bias[prev_keep_index]

    new_bn_layer = nn.BatchNorm2d(len(prev_keep_index))
    new_bn_layer.weight.data = bn_weights
    new_bn_layer.bias.data = bn_bias

    return new_bn_layer, prev_merge_pairs

def backward_linear_prune(linear_block, thresh):
    prev_keep_index = None
    new_modules = []
    for module in reversed(linear_block):
        if isinstance(module, nn.Linear):
            new_linear_layer, prev_keep_index = update_linear_structure(module, prev_keep_index, thresh)
            new_modules.insert(new_linear_layer)
        if isinstance(module, nn.ReLU):
            new_modules.insert(module)
    new_fc = nn.Sequential(*new_modules)
    return new_fc
        

def prune_model(model, thresh):
    # not done
    layer = 0
    prev_merge_pairs = None
    module_names = []
    new_modules = []
    current_layer = []
    layer = 0
    for name, module in model.named_modules():
        if 'conv' in name:
            module_names.append(name)
            new_mod, prev_merge_pairs = update_conv_structure(module, prev_merge_pairs, thresh)
        elif 'bn' in name:
            module_names.append(name)
            new_mod, _ = update_bn_structure(module, prev_merge_pairs)
            module = new_bn_layer
        elif name.endswith('downsample'):
            module_names.append(name)
            new_conv_layer, new_bn_layer = update_residual_connection_structure(module, prev_merge_pairs)
            new_mod = nn.Sequential(new_conv_layer, new_bn_layer)
        elif name == 'fc' and module_names[-1] != 'fc':
            new_mod = backward_linear_prune(module, thresh)
        
        if 'layer' in name:
            layer_num = int(name[name.find('layer') + 5])
            if layer_num == layer:
                current_layer.append(new_mod)
            else:
                new_block = BasicBlock(*current_layer)
                new_modules.append(new_block)
                layer_num += 1
                current_layer = []
        elif len(module_names) >= 1 and module_names[-1] != 'fc':
            module_names.append(name)
            new_modules(new_mod)

    for i in range(len(module_names)):
        setattr(model, module_names[i], new_modules[i])
    return model

    
        
