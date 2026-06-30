import logging
import torch

logger = logging.getLogger(__name__.split(".")[-1])

def load_model(model, MODEL_PATH, compile_model, device):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    if (device == torch.device("cuda")) and compile_model:
        try:
            model = torch.compile(model)
            print("Model compiled")
        except Exception as e:
            print(f"Compile skipped: {e}")

    logger.info(f"Model loaded successfully from {MODEL_PATH}")

def save_model(model, path):
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(raw_model.state_dict(), path)
    logger.info(f"Current model saved to directory {path}")
