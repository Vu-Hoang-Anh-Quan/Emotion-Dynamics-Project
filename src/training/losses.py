import math

import torch
import torch.nn as nn

def compute_erc_loss(logits, labels, weights=None):
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

def compute_attention_loss(attention_probs, utterance_mask):
    """
    attention_probs: [B, T, T]
    utterance_mask: [B, T]

    Returns:
        loss: scalar
    """

    total_loss = 0.0

    for b in range(attention_probs.shape[0]):
        valid_queries = utterance_mask[b].bool() # [T]
        probs = attention_probs[b][valid_queries] # [N_valid, T]

        # Renyi entropy of order 2
        h2 = -torch.log(
            probs.square().sum(dim=-1).clamp(min=1e-12)
        ) # [N_valid]

        target = math.log(2.0)

        loss = ((h2 - target) ** 2).mean()

        total_loss += loss

    return total_loss/attention_probs.shape[0]

def compute_loss(output, labels, weights=None, run_config=None):
    logits = output["logits"]
    erc_loss = compute_erc_loss(logits, labels, weights=weights)

    if ("use_attention_reg" in run_config and run_config["use_attention_reg"] == True) and run_config["attention_reg_rate"] > 0:
        attention_probs = output["attention_probs"]
        utterance_mask = output["utterance_mask"]
        attention_loss = compute_attention_loss(attention_probs, utterance_mask)
    else:
        attention_loss = 0.0

    total_loss = erc_loss + run_config["attention_reg_rate"] * attention_loss
    return total_loss

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