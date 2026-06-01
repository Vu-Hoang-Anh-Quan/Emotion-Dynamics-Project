import torch
from tqdm import tqdm
from .device import setup_device
from .optimizer import get_optimizer
from .losses import compute_loss
from .losses import compute_class_weights
from .metrics import evaluate
from .checkpoint import save_model

def train_one_epoch(model, dataloader, optimizer, loss_function, device, use_amp, scaler):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        utterance_mask = batch["utterance_mask"].to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(input_ids, attention_mask, utterance_mask=utterance_mask)
                loss = loss_function(logits, labels)

            if (torch.isnan(loss)):
                print("Loss is already NaN here, before propagating back")

            scaler.scale(loss).backward()
            
            # Avoid gradients explosion
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

        else:
            logits = model(input_ids, attention_mask, utterance_mask=utterance_mask)
            loss = loss_function(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Avoid gradient explosion
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train_model(model, train_loader, val_loader, config, model_path):
    # logger = load_logging_system()

    device, use_amp, scaler = setup_device(config)

    print(f"Using device: {device} | AMP: {use_amp}")

    # Optional compile (safe guard)
    if config["use_cuda"] and config["compile_model"]:
        try:
            model = torch.compile(model)
            print("Model compiled")
        except Exception as e:
            print(f"Compile skipped: {e}")
    
    # Optimizer here
    optimizer = get_optimizer(model, config)

    # Get your class_weights
    class_weights = compute_class_weights(train_loader, num_classes=config["dataset"][config["dataset_name"]]["num_labels"], device=device)
    print(class_weights)
    # Your custom loss function
    loss_function = lambda logits, labels: compute_loss(logits, labels, weights=class_weights)

    if (config["debug"]): 
        # Can add overfit one batch here if necessary
        torch.autograd.set_detect_anomaly(True)

    best_f1 = 0

    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        # logger.info(f"Epoch {epoch+1}/{config['epochs']}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_function, device, use_amp, scaler
        )

        val_loss, val_acc, val_f1_m, val_f1_m_ex = evaluate(
            model, val_loader, loss_function, device
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1-score macro: {val_f1_m:.4f} | Val F1-score macro non-Neutral: {val_f1_m_ex:.4f}")

        # logger.info(f"Train Loss: {train_loss:.4f}")
        # logger.info(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1-score macro: {val_f1_m:.4f} | Val F1-score macro non-Neutral: {val_f1_m_ex:.4f}")

        # Debug nan
        # debug_nan(model)

        if val_f1_m_ex >= best_f1:
            best_f1 = val_f1_m_ex
            save_model(model=model, path=model_path)
            # Save model