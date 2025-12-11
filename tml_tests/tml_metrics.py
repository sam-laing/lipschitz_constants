from .fgsm import test_fgsm
from .load_ood import get_svhn_loader 
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch
from collections import OrderedDict  
import numpy as np

from typing import Dict, Any



def evaluate_model(model, test_loader, cfg, device):
    """    
    given the model and test loader, return a dict of important metrics for the model 
    """
 
    outputs, labels = get_model_outputs_labels(model, test_loader)


    critierion = torch.nn.CrossEntropyLoss()
    loss = critierion(outputs, labels)

    #get the predicted probabilities
    outputs_prob = torch.softmax(outputs, dim=1).cpu().detach().numpy()

    #convert labels to numpy array
    labels_one_hot = labels.cpu().numpy()


    ood_loader = get_svhn_loader(subset_size=5_000, random_seed=cfg.seed)
    ood_auroc = do_ood_detection(ood_loader, test_loader, model, device)

    fgsm = test_fgsm(test_loader, model, device, eps=0.15)

    return {
        "ood": ood_auroc, "fgsm": fgsm, 
    }










# import sklearn auroc
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F


def do_ood_detection(ood_loader, test_loader, model, device):
    """
    Perform OOD detection by computing the AUROC
    with max softmax probability and ood data from SVHN
    """
    model.eval()

    test_outputs = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            outs = model(data)
            outs = F.softmax(outs, dim=-1)
            test_outputs.append(outs.cpu())
    if len(test_outputs) == 0:
        return float("nan")
    test_outputs = torch.cat(test_outputs, dim=0)

    ood_outputs = []
    with torch.no_grad():
        for data, target in ood_loader:
            data = data.to(device)
            outs = model(data)
            outs = F.softmax(outs, dim=-1)
            ood_outputs.append(outs.cpu())
    if len(ood_outputs) == 0:
        return float("nan")
    ood_outputs = torch.cat(ood_outputs, dim=0)

    output_tensor = torch.cat((test_outputs, ood_outputs), dim=0)
    n_test = test_outputs.size(0)
    n_ood = ood_outputs.size(0)

    binary_label_tensor = torch.cat((
        torch.ones(n_test, dtype=torch.long),
        torch.zeros(n_ood, dtype=torch.long)
    ), dim=0)
    binary_label_array = binary_label_tensor.numpy()

    max_outputs = torch.max(output_tensor, dim=1)[0].numpy()
    auroc = roc_auc_score(binary_label_array, max_outputs)

    return auroc



def _get_accuracy(outputs, labels):
    return (100*(torch.argmax(labels,1) == torch.argmax(outputs, 1)).sum() / labels.shape[0]).item()

def _get_precision_and_recall(outputs, labels):
    _, predicted_labels = torch.max(outputs, 1)
    true_labels = torch.argmax(labels, 1)

    predicted_labels_np = predicted_labels.cpu().numpy()
    true_labels_np = true_labels.cpu().numpy()

    precision = precision_score(true_labels_np, predicted_labels_np, average='macro')
    recall = recall_score(true_labels_np, predicted_labels_np, average='macro')

    return precision, recall


def _get_bins(outputs: torch.tensor, labels: torch.tensor, n_bins=10):
    '''
    Computes the Expected Calibration Error (ECE).
    outputs: (n_samples, n_classes) already passed through softmax
    labels: (n_samples, n_classes) as indices of the true classes

    returns: bins_pred, bins_y, bin_sizes
    '''
    assert outputs.shape[0] == labels.shape[0]
    N = outputs.shape[0]
    probs = outputs
    #labels = torch.argmax(labels, axis=1)
    corrects = (torch.argmax(probs, axis=1) == labels).float()
    corrects = corrects.unsqueeze(1)
    # just need the model's confidence of the true class 
    pred_conf = torch.gather(probs, 1, labels.unsqueeze(1))
    bin_sizes = []
    confs = []
    accs = []
    for i in range(n_bins):
        bin_min = i / n_bins
        bin_max = (i + 1) / n_bins

        B = torch.where((pred_conf > bin_min) & (pred_conf <= bin_max))
        size = B[0].shape[0]
        bin_sizes.append(size)
        confs.append(torch.mean(pred_conf[B]).item())
        accs.append(torch.mean(corrects[B]).item())

    return confs, accs, bin_sizes


def get_ece_and_overconf_err(outputs, labels, num_bins = 10):
    assert outputs.shape[0] == labels.shape[0], "outputs and labels must have the same number of samples"
    N = outputs.shape[0]
    bins_pred, bins_y, bin_sizes = _get_bins(outputs, labels, num_bins)

    ece = 0
    oe = 0
    for conf, acc, size in zip(bins_pred, bins_y, bin_sizes):
        if size != 0:
            ece += size * abs(conf-acc)
            oe += size * (conf * max(conf - acc, 0))

    return ece /N, oe/N

def reliablity_diagram(outputs, labels, num_bins=10):
    """
    returns the reliability diagram of the model
    """
    import matplotlib.pyplot as plt

    bins_pred, bins_y, bin_sizes = _get_bins(outputs, labels, num_bins)
    
    accuracies = np.zeros(num_bins)
    confidences = np.zeros(num_bins)
    
    for i in range(num_bins):
        if bin_sizes[i] > 0:
            accuracies[i] = np.mean(bins_y[i])
            confidences[i] = np.mean(bins_pred[i])
    
    # Plot the reliability diagram
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.plot(confidences, accuracies, marker='o', label='Model')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid()
    plt.show()

    return accuracies, confidences


    

    

def ood_uncertainty(model, device):
    """
    given the model and ood_loader, return the aleatoric uncertainty and epistemic uncertainty
    """
    svhn_loader = get_svhn_loader()
    model.eval()

    outputs, labels = get_model_outputs_labels(svhn_loader, model, device)

    return outputs, labels


def get_model_outputs_labels(model, test_loader):
    """
    Get model outputs and true labels from the test loader.
    """
    model.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(next(model.parameters()).device)
            outputs = model(data)
            all_outputs.append(outputs.cpu())
            all_labels.append(target.cpu())
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    
    return all_outputs, all_labels