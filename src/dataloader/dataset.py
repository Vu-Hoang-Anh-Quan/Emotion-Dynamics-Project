# import torch
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "utterances": item["utterances"],
            "labels": item["emotion"]
        }