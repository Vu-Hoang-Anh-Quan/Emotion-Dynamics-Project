from torch.utils.data import DataLoader
from .dataset import EmotionDataset

def build_dataloaders(tokenized_data, batch_size, do_shuffling):
    dataset = EmotionDataset(tokenized_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=do_shuffling)

    return dataloader