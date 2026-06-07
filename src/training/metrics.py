from sklearn.metrics import f1_score
from collections import Counter
import torch
from .losses import compute_loss

def evaluate(model, dataloader, loss_function, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            # Move everything to device
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            labels = batch["labels"]

            logits = model(batch)
            loss = loss_function(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)

            # Find out the total using mask
            mask = labels != -100
            correct += ((preds == labels)&mask).sum().item()
            total += mask.sum().item()

            # 🔹 store for F1
            all_preds.extend(preds[mask].cpu().tolist())
            all_labels.extend(labels[mask].cpu().tolist())

    acc = correct / total if total > 0 else 0

    # Check percentage
    counts = Counter(all_preds)
    for i in range(7):
        count = counts[i]
        percentage = (count / len(all_preds))*100
        print(f"Class {i}: {percentage:.4f}") # Remove this later, as tmux printing is not sth you should rely on

    # conpute F1 macro
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    # 🔹 compute F1 without Neutral layer (standard for DailyDialog)
    f1_macro_excluding_neutral = f1_score(
        all_labels, 
        all_preds, 
        labels=[1, 2, 3, 4, 5, 6],
        average='macro'
    )

    return total_loss / len(dataloader), acc, f1_macro, f1_macro_excluding_neutral

def get_final_test_accuracy(model, test_loader, device):
    loss_function = compute_loss
    test_loss, test_accuracy, test_f1_m, test_f1_m_ex = evaluate(model, test_loader, loss_function, device)
    return test_loss, test_accuracy, test_f1_m, test_f1_m_ex