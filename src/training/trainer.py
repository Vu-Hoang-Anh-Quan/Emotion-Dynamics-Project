import torch
import torch.nn as nn
from tqdm import tqdm
from .debug import debug_overfit_one_batch
import logging
from sklearn.metrics import f1_score
from collections import Counter

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

def compute_loss(logits, labels, weights=None):
    # Expecting shape [B, T, num_labels] and [B, T]
    B, T, C = logits.shape

    logits = logits.view(B*T, C)
    labels = labels.view(B*T)

    loss_function = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    return loss_function(logits, labels)

def compute_class_weights(loader, num_classes, device):
    counts = torch.zeros(num_classes, dtype=torch.float)
    for batch in loader:
        labels = batch["labels"] # [B, T]

        labels = labels.view(-1) # [B * T]
        lables = labels[labels != -100] # Exclude all padded labels

        counts += torch.bincount(lables, minlength=num_classes)
    
    # Ensure everything is at least 1, avoid dividing by 0
    counts = torch.clamp(counts, min=1)
    # Inverse probability
    weights = counts.sum() / counts
    # Normalize
    weights = weights / weights.mean()

    return weights.to(device)

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

            preds = torch.argmax(logits, dim=2)

            # Find out the total using mask
            mask = labels != -100
            correct += ((preds == labels)&mask).sum().item()
            total += mask.sum().item()

            # 🔹 store for F1
            all_preds.extend(preds[mask].cpu().tolist())
            all_labels.extend(labels[mask].cpu().tolist())

    acc = correct / total

    # Check percentage
    counts = Counter(all_preds)
    for i in range(7):
        count = counts[i]
        percentage = (count / len(all_preds))*100
        print(f"Class {i}: {percentage:.4f}")

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

def get_optimizer(model, config):
    # [bert_or_head][decay_or_no_decay]
    # 0 = head, 1 = bert
    # 0 = decay, 1 = no_decay
    separated_params = [[[] for _ in range(2)] for _ in range(2)]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # print(name)
        separated_params["bert" in name]["bias" in name or "LayerNorm" in name].append(param)

    optimizer_grouped_parameters = []
    for i in range(2):
        for j in range(2):
            params = separated_params[i][j]
            if not params: continue
            
            optimizer_grouped_parameters.append({
                "params": params,
                "weight_decay": 0.0 if j == 1 else config["weight_decay"],
                "lr": config["lr_bert"] if i == 1 else config["lr_head"]
            })

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    # for group in optimizer.param_groups:
    #     print(len(group["params"]))
    return optimizer

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
    
    decay = []
    no_decay = []

    # Optimizer here
    optimizer = get_optimizer(model, config)

    # Get your class_weights
    class_weights = compute_class_weights(train_loader, num_classes=config["num_labels"], device=device)
    print(class_weights)
    # Your custom loss function
    loss_function = lambda logits, labels: compute_loss(logits, labels, weights=class_weights)

    if (config["debug"]): 
        debug_overfit_one_batch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_function,
            device=device,
            # steps=config["epochs"]
        )
        return 

    best_f1 = 0

    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        logger.info(f"\nEpoch {epoch+1}/{config['epochs']}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_function, device, use_amp, scaler
        )

        val_loss, val_acc, val_f1_m, val_f1_m_ex = evaluate(
            model, val_loader, loss_function, device
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1-score macro: {val_f1_m:.4f} | Val F1-score macro non-Neutral: {val_f1_m_ex:.4f}")

        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1-score macro: {val_f1_m:.4f} | Val F1-score macro non-Neutral: {val_f1_m_ex:.4f}")

        if val_f1_m_ex >= best_f1:
            best_f1 = val_f1_m_ex
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(raw_model.state_dict(), model_path)
            print(f"Current model saved to directory {model_path}")
            logger.info(f"Current model saved to directory {model_path}")

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
    loss_function = compute_loss
    test_loss, test_accuracy, test_f1_m, test_f1_m_ex = evaluate(model, test_loader, loss_function, device)
    return test_loss, test_accuracy, test_f1_m, test_f1_m_ex