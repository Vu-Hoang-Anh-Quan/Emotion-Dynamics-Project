from torch.utils.data import DataLoader
from .dataset import EmotionDataset
from transformers import DataCollatorWithPadding, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def sort_key(example):
    return len(example["input_ids"])

def build_dataloaders(tokenized_data, batch_size, do_shuffling):
    dataset = EmotionDataset(tokenized_data)

    # Sort for better performance
    if not do_shuffling:
        dataset.data.sort(key=sort_key)

    collate_fn = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True  # pad to longest in batch
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=do_shuffling, collate_fn=collate_fn)

    return dataloader