import torch

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