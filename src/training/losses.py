import torch
import torch.nn as nn

def compute_loss(logits, labels, weights=None):
    # Detect problem
    if logits.dim() > 3:
        raise RuntimeError(f"Logits have {logits.dim()} dimensions, expecting 2 or 3")
    # If input is [B, T, C] then flatten it
    if logits.dim() == 3:
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        labels = labels.view(B*T)

    loss_function = nn.CrossEntropyLoss(weight=weights, ignore_index=-100) # Each instance create a new function
    return loss_function(logits, labels)

def compute_class_weights(loader, num_classes, device):
    counts = torch.zeros(num_classes, dtype=torch.float)
    for batch in loader:
        labels = batch["labels"] # [B, T]

        labels = labels.view(-1) # [B * T]
        labels = labels[labels != -100] # Exclude all padded labels

        counts += torch.bincount(labels, minlength=num_classes) # will break if labels ever move to GPU
    
    # Ensure everything is at least 1, avoid dividing by 0
    counts = torch.clamp(counts, min=1)
    # Inverse probability
    weights = counts.sum() / counts
    # Normalize
    weights = weights / weights.mean()

    return weights.to(device)