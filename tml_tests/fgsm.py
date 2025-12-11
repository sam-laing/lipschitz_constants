import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

def fgsm_attack_normalized(imgs, data_grads, epsilon=0.03):
    """
    Perform FGSM attack on normalized images.
    """
    sign_data_grads = data_grads.sign()
    perturbed_imgs = imgs + epsilon * sign_data_grads
    perturbed_imgs = torch.clamp(perturbed_imgs, -2.5, 2.5)
    return perturbed_imgs

def normalize(x, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(x.device)
    return (x - mean) / std

def denormalize(x, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(x.device)
    return x * std + mean

def test_fgsm(test_loader, model, device, eps=0.03):
    model.eval()
    init_correct = 0
    correct = 0
    total = 0

    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        
        # Get initial predictions
        with torch.no_grad():
            outputs = model(data)
            init_pred = outputs.max(1, keepdim=True)[1]
        
        # Compute gradients for attack
        data.requires_grad = True
        outputs = model(data)
        loss = F.cross_entropy(outputs, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        
        # Perform FGSM
        perturbed_data = fgsm_attack_normalized(data, data_grad, epsilon=eps)
        
        # Get predictions on perturbed data
        with torch.no_grad():
            outputs_perturbed = model(perturbed_data)
            final_pred = outputs_perturbed.max(1, keepdim=True)[1]
        
        total += target.size(0)
        init_correct += init_pred.eq(target.view_as(init_pred)).sum().item()
        correct += final_pred.eq(target.view_as(final_pred)).sum().item()

    new_acc = correct / total
    init_acc = init_correct / total
    acc_diff = new_acc - init_acc
    return new_acc, init_acc, acc_diff



