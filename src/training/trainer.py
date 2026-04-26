import torch
import torch.nn as nn
from tqdm import tqdm
from .debug import debug_overfit_one_batch
import logging
from sklearn.metrics import f1_score

def load_logging_system():
    logger = logging.getLogger(__name__)
    return logger

def setup_device(config):
    use_cuda = config["use_cuda"]

    if use_cuda:
        if torch.cuda.is_available(): device = torch.device("cuda")
        else: raise RuntimeError("Config is set to use CUDA, yet no CUDA available")
    else: device = torch.device("cpu")
    

    # AMP only for GPU
    use_amp = use_cuda and config.get("use_amp", True)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    return device, use_amp, scaler 

def train_one_epoch(model, dataloader, optimizer, loss_function, device, use_amp, scaler):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(input_ids, attention_mask)
                loss = loss_function(logits, labels)

            scaler.scale(loss).backward()
            
            # Avoid gradients explosion
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

        else:
            logits = model(input_ids, attention_mask)
            loss = loss_function(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Avoid gradient explosion
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, loss_function, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

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

            # 🔹 store for F1
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / total

    # 🔹 compute F1 without Neutral layer (standard for DailyDialog)
    f1_macro = f1_score(
        all_labels, 
        all_preds, 
        labels=[1, 2, 3, 4, 5, 6],
        average='macro'
    )

    return total_loss / len(dataloader), acc, f1_macro

def train_model(model, train_loader, val_loader, config, model_path):
    logger = load_logging_system()

    device, use_amp, scaler = setup_device(config)

    print(f"Using device: {device} | AMP: {use_amp}")

    # Optional compile (safe guard)
    if config["use_cuda"]:
        try:
            model = torch.compile(model)
            print("Model compiled")
        except Exception as e:
            print(f"Compile skipped: {e}")

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

    best_val_loss = float("inf")

    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_function, device, use_amp, scaler
        )

        val_loss, val_acc, val_f1_m = evaluate(
            model, val_loader, loss_function, device
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1-score macro: {val_f1_m:.4f}")

        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1-score macro: {val_f1_m:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(raw_model.state_dict(), model_path)
            print(f"Current model saved to directory {model_path}")

def load_model(model, MODEL_PATH):
    use_cuda = torch.cuda.is_available()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cuda" if use_cuda else "cpu"))
    if use_cuda:
        try:
            model = torch.compile(model)
            print("Model compiled")
        except Exception as e:
            print(f"Compile skipped: {e}")


def get_final_test_accuracy(model, test_loader, device):
    loss_function = torch.nn.CrossEntropyLoss()
    test_loss, test_accuracy, test_f1_m = evaluate(model, test_loader, loss_function, device)
    return test_loss, test_accuracy, test_f1_m