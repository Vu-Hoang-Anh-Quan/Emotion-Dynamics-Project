import os
import torch

def save_data(data, save_path):
    # Due to dynamic padding in each batch, you can actually sort them by utterances length for further running optimization
    # However, you should shuffle train_data
    # ensure folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(data, save_path)
    print(f"Saved to {save_path}")