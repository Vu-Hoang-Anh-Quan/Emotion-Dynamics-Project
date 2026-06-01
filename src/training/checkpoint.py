import torch

def load_model(model, MODEL_PATH, compile_model):
    use_cuda = torch.cuda.is_available()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cuda" if use_cuda else "cpu"))
    if use_cuda and compile_model:
        try:
            model = torch.compile(model)
            print("Model compiled")
        except Exception as e:
            print(f"Compile skipped: {e}")

def save_model(model, path):
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(raw_model.state_dict(), path)
    print(f"Current model saved to directory {path}")
    # logger.info(f"Current model saved to directory {path}")