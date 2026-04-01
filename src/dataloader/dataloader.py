from torch.utils.data import DataLoader
from .dataset import EmotionDataset


def build_dataloaders(train_tokenized, val_tokenized, test_tokenized, batch_size=16):
    train_dataset = EmotionDataset(train_tokenized)
    val_dataset   = EmotionDataset(val_tokenized)
    test_dataset  = EmotionDataset(test_tokenized)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader