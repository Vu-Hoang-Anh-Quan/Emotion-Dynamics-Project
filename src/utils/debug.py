import torch
import math
import numpy

def attention_outputs_cosine_similarity(attention_scores_list):
    num_layers = len(attention_scores_list)
    B = len(attention_scores_list[0]) # [num_layers, B, T, H]

    print("\nCosine similarity of attention scores")
    for i in range(num_layers):
        for j1 in range(3):
            for j2 in range(j1+1, 4):
                cos_val = torch.nn.functional.cosine_similarity(
                    numpy.array(attention_scores_list[j1]).reshape(B, -1).tolist(),
                    numpy.array(attention_scores_list[j2]).reshape(B, -1).tolist(),
                    dim = -1
                )
                print(f"{cos_val:.3f}", end=" ")
        print()

def cal_jensen_shannon(probs1, probs2):
    eps = 1e-8

    p = probs1.clamp(min=eps)
    q = probs2.clamp(min=eps)

    m = 0.5 * (p + q)

    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)

    js = 0.5 * (kl_pm + kl_qm)

    return js.mean().item()

def attention_probs_jensen_shannon(attention_probs_list):
    print("\nJensen-Shannon divergence of attention probabilities")
    for probs_dict in attention_probs_list:
        list = []
        for probs in probs_dict.values():
            list.append(probs)
        for j1 in range(3):
            for j2 in range(j1+1, 4):
                jensen_val = cal_jensen_shannon(list[j1], list[j2])
                print(f"{jensen_val:.3f}", end=" ")
        print()

def debug_nan(model):
    print("NaN parameters:\n")
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(name)
    print("\n")

def list_bad_parameters_if_exist(model):
    nan_parameters = []
    inf_parameters = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_parameters.append(name)
        if torch.isinf(param).any():
            inf_parameters.append(name)        

    if nan_parameters or inf_parameters: # Check if the list has something in it
        print("\nNaN parameters:")
        for name in nan_parameters:
            print(f"{name}")
        print("\nINF parameters:")
        for name in inf_parameters:
            print(f"{name}")
        raise RuntimeError("BAD PARAMETERS")
    
def check_bad_gradient(model):
    bad_gradient = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                bad_gradient.append(name)

    if bad_gradient: 
        print("\nBAD GRADIENTS:")
        for name in bad_gradient:
            print(f"{name}")
        # raise RuntimeError("BAD GRADIENTS")

def inspect_attention_probability(attention_probs, utterance_mask):
    torch.set_printoptions(precision=3, sci_mode=False)
    # [B, T, T]
    B, T, _ = attention_probs.shape
    for i in range(B):
        print(f"\nConversation {i} in batch\n")
        for j in range(T):
            if utterance_mask[i][j] == 0:
                continue
            for k in range(T):
                if utterance_mask[i][k] == 0:
                    continue
                print(f"{attention_probs[i][j][k].item():.2f}", end=" ")
            print()
            
    torch.set_printoptions(profile='default') # Resets it back to normal

def check_tensor(name, x):
    if torch.isnan(x).any():
        print(f"{name}: NaN")
    if torch.isinf(x).any():
        print(f"{name}: Inf")

    print(
        f"{name}: "
        f"min={x.min().item():.4f}, "
        f"max={x.max().item():.4f}, "
        f"mean={x.mean().item():.4f}"
    )

def debug_overfit_one_batch(model, dataloader, optimizer, loss_fn, device, steps=100):
    model.train()

    batch = next(iter(dataloader))

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    # Mask for valid tokens
    mask = labels != -100

    print("Valid tokens:", mask.sum().item())
    print("Unique labels:", torch.unique(labels[mask]))

    for step in range(steps):
        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)

        loss.backward()

        # 🔍 Gradient check (very important)
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item()

        optimizer.step()

        # 🔍 Predictions
        preds = torch.argmax(logits, dim=2)

        correct = ((preds == labels) & mask).sum().item()
        total = mask.sum().item()
        acc = correct / total if total > 0 else 0

        if step % 10 == 0 or step == steps - 1:
            print(
                f"Step {step:03d} | "
                f"Loss: {loss.item():.4f} | "
                f"Acc: {acc:.4f} | "
                f"GradNorm: {total_grad_norm:.4f}"
            )