import logging
import torch
from tqdm import tqdm
from .device import setup_device
from .optimizer import get_optimizer
from .losses import compute_loss
from .losses import compute_class_weights
from .metrics import evaluate
from .checkpoint import save_model

logger = logging.getLogger(__name__.split(".")[-1])

def train_one_epoch(model, dataloader, optimizer, loss_function, device, use_amp, scaler):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        # Move everything to device
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        labels = batch["labels"]

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(batch)
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
            logits = model(batch)
            loss = loss_function(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Avoid gradient explosion
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train_model(model, train_loader, val_loader, config, running_pipeline, model_path, device, use_amp, scaler):
    logger.info(f"Using device: {device} | AMP: {use_amp}")

    # Optional compile (safe guard)
    if config["use_cuda"] and config["compile_model"]:
        try:
            model = torch.compile(model)
            logger.info("Model compiled")
        except Exception as e:
            logger.warning(f"[WARNING] Compile skipped: {e}")
    
    # Optimizer here
    optimizer = get_optimizer(model, config)

    # Get your class_weights
    class_weights = compute_class_weights(train_loader, num_classes=config["dataset"][config["dataset_name"]]["num_labels"], device=device)
    logger.info(f"Class weights: {class_weights}")
    # Your custom loss function
    loss_function = lambda logits, labels: compute_loss(logits, labels, weights=class_weights)

    if (config["debug"]): 
        # Can add overfit one batch here if necessary
        torch.autograd.set_detect_anomaly(True)
    else:
        torch.autograd.set_detect_anomaly(False)

    best_f1 = 0

    for epoch in range(config[running_pipeline]["epochs"]):
        logger.info(f"Epoch {epoch+1}/{config[running_pipeline]['epochs']}")
        # logger.info(f"Epoch {epoch+1}/{config['epochs']}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_function, device, use_amp, scaler
        )

        val_loss, val_acc, val_f1_m, val_f1_m_ex = evaluate(
            model, val_loader, loss_function, device
        )

        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1-score macro: {val_f1_m:.4f} | Val F1-score macro non-Neutral: {val_f1_m_ex:.4f}")

        # logger.info(f"Train Loss: {train_loss:.4f}")
        # logger.info(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1-score macro: {val_f1_m:.4f} | Val F1-score macro non-Neutral: {val_f1_m_ex:.4f}")

        # Debug nan
        # debug_nan(model)

        if val_f1_m_ex >= best_f1:
            best_f1 = val_f1_m_ex
            save_model(model=model, path=model_path)
            # Save model