import torch

def debug_overfit_one_batch(model, dataloader, optimizer, loss_fn, device, steps=50):
    model.train()

    batch = next(iter(dataloader))

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    for i in range(steps):
        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        print(f"Step {i}: loss = {loss.item():.4f}")