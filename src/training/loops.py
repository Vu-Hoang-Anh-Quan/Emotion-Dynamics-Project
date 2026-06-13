import logging
import torch
from tqdm import tqdm
from .optimizer import get_optimizer
from .losses import compute_loss
from .losses import compute_class_weights
from .metrics import evaluate
from .checkpoint import save_model
from ..utils.debug import list_bad_parameters_if_exist, check_bad_gradient

logger = logging.getLogger(__name__.split(".")[-1])

def run_epoch_hooks(model, optimizer, config, pipeline_config, epoch):
    anything_changed = False
    if epoch == 0 and pipeline_config["bert_freeze_epochs"] > 0:
        model.embedding.set_trainable_layers(k=0)
        anything_changed = True
        logger.info("Freeze BERT")

    if epoch == pipeline_config["bert_freeze_epochs"]:
        model.embedding.set_trainable_layers(k=pipeline_config["bert_trainable_layer"])
        anything_changed = True
        logger.info(f"Set last {pipeline_config['bert_trainable_layer']} of BERT trainable")

    # If anything changed -> get optimizer again
    if anything_changed:
        optimizer = get_optimizer(model, config)
        # optimizer.zero_grad()

    return optimizer

def train_one_epoch(model, dataloader, optimizer, loss_function, device, use_amp, scaler, debug = False):
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

                if not torch.isfinite(logits).all():
                    print("BAD LOGITS")


                loss = loss_function(logits, labels)

            if not torch.isfinite(loss):
                print(f"Loss is NOT finite: {loss}")

            scaler.scale(loss).backward()
            
            # Avoid gradients explosion
            scaler.unscale_(optimizer)

            check_bad_gradient(model)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            old_scale = scaler.get_scale()

            scaler.step(optimizer)
            scaler.update()

            new_scale = scaler.get_scale()
            if new_scale < old_scale:
                print(f"Overflow detected: {old_scale} -> {new_scale}")

            # Check if the model has any NaN parameters
            if debug: list_bad_parameters_if_exist(model)

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

        optimizer = run_epoch_hooks(
            model=model,
            optimizer=optimizer,
            config=config,
            pipeline_config=config[running_pipeline],
            epoch=epoch
        )

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_function, device, use_amp, scaler,
            debug=config["debug"]
        )

        val_loss, val_acc, val_f1_m, val_f1_m_ex = evaluate(
            model, val_loader, loss_function, device
        )

        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1-score macro: {val_f1_m:.4f} | Val F1-score macro non-Neutral: {val_f1_m_ex:.4f}")

        # logger.info(f"Train Loss: {train_loss:.4f}")
        # logger.info(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1-score macro: {val_f1_m:.4f} | Val F1-score macro non-Neutral: {val_f1_m_ex:.4f}")

        # Debug nan
        # debug_nan(model)

        if val_f1_m_ex >= best_f1:
            best_f1 = val_f1_m_ex
            save_model(model=model, path=model_path)
            # Save model