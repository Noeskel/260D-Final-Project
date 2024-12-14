import torch
from torch import nn
from torch.nn import functional as F
import time

def prune_data(model, data, original_data_size, prune_percent=0.5, prev_acc=-1, device='cuda'):
    #start = time.time()
    model.eval()
    correctly_labeled_scores = []
    correctly_labeled_indexes = []
    incorrectly_labeled_scores = []
    incorrectly_labeled_indexes = []

    keep_indexes = []
    
    count_correct = 0
    total = 0
    dataloader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=False)
    for img, label, indexes in dataloader:
        img, label, indexes = img.to(device), label.to(device), indexes.to(device)

        pred = model(img)
        probs = F.softmax(pred.detach(), dim=-1)
        prob_score = probs.gather(1, label[None, :]).flatten()
        
        diversity_score = torch.sqrt(torch.sum((probs-0.1)**2, dim=-1))
        
        correctly_labeled = (torch.argmax(probs, dim=-1) == label).float()*2
        correctly_labeled -= 1

        count_correct += torch.count_nonzero(correctly_labeled > 0)
        total += len(correctly_labeled)
        
        constant = 2.0
        scores = prob_score + constant*diversity_score*correctly_labeled

        correctly_labeled_scores.append(scores[correctly_labeled == 1])
        correctly_labeled_indexes.append(indexes[correctly_labeled == 1])
        
        incorrectly_labeled_scores.append(scores[correctly_labeled == -1])
        incorrectly_labeled_indexes.append(indexes[correctly_labeled == -1])
    
    acc = count_correct/total

    if prev_acc > 0 and acc-prev_acc < 0.1:
        k_correct = len(correctly_labeled)
        k_incorrect = int(original_data_size*prune_percent)-k_correct
    else:
        k_correct = int(original_data_size*prune_percent*acc)
        k_incorrect = int(original_data_size*prune_percent*(1-acc))

    correctly_labeled_scores = torch.cat(correctly_labeled_scores)
    correctly_labeled_indexes = torch.cat(correctly_labeled_indexes)
    incorrectly_labeled_scores = torch.cat(incorrectly_labeled_scores)
    incorrectly_labeled_indexes = torch.cat(incorrectly_labeled_indexes)

    discrepency_correct = len(correctly_labeled_indexes) - k_correct
    discrepency_incorrect = len(incorrectly_labeled_indexes) - k_incorrect

    if discrepency_correct < 0:
        k_incorrect -= discrepency_correct
        k_correct = len(correctly_labeled_indexes)
    elif discrepency_incorrect < 0:
        k_correct -= discrepency_incorrect
        k_incorrect = len(incorrectly_labeled_indexes)

    _, correct_indexes = torch.topk(correctly_labeled_scores, k=k_correct,largest=False)
    _, incorrect_indexes = torch.topk(incorrectly_labeled_scores, k=k_incorrect,largest=True)
    keep_indexes += correctly_labeled_indexes[correct_indexes].tolist()
    keep_indexes += incorrectly_labeled_indexes[incorrect_indexes].tolist()
    #end = time.time()
    #print('prune time: ', start-end)
    
    return keep_indexes, acc