# import torch
from torch.utils.data import Dataset

class UtteranceEmotionDataset(Dataset):
    def __init__(self, data):
        self.samples = []

        for conversation in data:
            utterances = conversation["utterances"]
            labels = conversation["labels"]

            for utt, label in zip(utterances, labels):

                if label is None:
                    continue

                self.samples.append({
                    "utterance": utt,
                    "label": label
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class EmotionDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "utterances": item["utterances"],
            "labels": item["labels"]
        }