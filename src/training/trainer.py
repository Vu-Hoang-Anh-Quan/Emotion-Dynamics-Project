import torch
import torch.nn as nn
from tqdm import tqdm
from .debug import debug_overfit_one_batch

def train_one_epoch(model, dataloader, optimizer, loss_function, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = loss_function(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, loss_function, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_function(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    return total_loss / len(dataloader), acc

def train_model(model, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"]
    )

    loss_function = nn.CrossEntropyLoss()

    if (config["debug"]): 
        debug_overfit_one_batch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_function,
            device=device,
            steps=config["epochs"]
        )
        return 


    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_function, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, loss_function, device
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.4f}")

def get_final_test_accuracy(model, test_loader, device):
    loss_function = torch.nn.CrossEntropyLoss()
    test_loss, test_accuracy = evaluate(model, test_loader, loss_function, device)
    return test_loss, test_accuracy