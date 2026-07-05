import torch
from torch.utils.data import DataLoader
from .dataset import EmotionDataset, UtteranceEmotionDataset
from transformers import AutoTokenizer

tokenizer: AutoTokenizer

def load_tokenizer():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def utterance_collate_fn(batch, max_len=512):
    global tokenizer

    utterances = [
        item["utterance"]
        for item in batch
    ]

    labels = torch.tensor([
        item["label"]
        for item in batch
    ])

    enc = tokenizer(
        utterances,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    return {
        "input_ids": enc["input_ids"],          # [B, L]
        "attention_mask": enc["attention_mask"],# [B, L]
        "labels": labels                        # [B]
    }

def build_utterance_dataloader(data, batch_size, do_shuffling):
    dataset = UtteranceEmotionDataset(data)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=do_shuffling,
        collate_fn=lambda batch: utterance_collate_fn(batch)
    )

    return dataloader

def conversation_collate_fn(batch, max_len=512):
    global tokenizer

    batch_utterances = [item["utterances"] for item in batch]
    batch_labels = [item["labels"] for item in batch]

    # Flatten and store size of each batch
    utt_len, flatten_utts = [], []
    for conversation in batch_utterances:
        utt_len.append(len(conversation))
        for utt in conversation:
            flatten_utts.append(utt)

    if (sum(utt_len) != len(flatten_utts)):
        raise RuntimeError("Sum of utterances length is different to the number of utterances after flattened")

    # Tokenize with dynamic padding
    enc = tokenizer(
        flatten_utts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    # Reshape
    idx = 0
    raw_input_ids, raw_attention_mask = [], []
    for l in utt_len:
        raw_input_ids.append(enc["input_ids"][idx:idx + l])
        raw_attention_mask.append(enc["attention_mask"][idx:idx + l])
        idx += l

    # Pad Turn_len
    #   Define T in result [Batch_len, Turn_len, Utt_len]
    max_turns = max(utt_len)
    padded_input_ids, padded_attention_mask = [], []
    for inp, att in zip(raw_input_ids, raw_attention_mask):
        current_turns, current_len = inp.shape
        pad_size = max_turns - current_turns

        if pad_size > 0:
            pad_ids = torch.zeros(pad_size, current_len, dtype=torch.long)
            pad_att = torch.zeros(pad_size, current_len, dtype=torch.long)

            inp = torch.cat([inp, pad_ids], dim=0)
            att = torch.cat([att, pad_att], dim=0)

        padded_input_ids.append(inp)
        padded_attention_mask.append(att)

    # Remember labels
    padded_labels = []
    padded_utterance_mask = []
    for labs in batch_labels:
        labels_tensor = torch.tensor([
            -100 if l is None else l for l in labs
        ])
        uttterance_mask_tensor = torch.tensor([
            0 if l is None else 1 for l in labs
        ])

        pad_size = max_turns - len(labs)

        if (pad_size > 0):
            labels_tensor = torch.cat([
                labels_tensor,
                torch.full((pad_size,), -100)
            ])
            uttterance_mask_tensor = torch.cat([
                uttterance_mask_tensor,
                torch.full((pad_size,), 0)
            ])
        padded_labels.append(labels_tensor)
        padded_utterance_mask.append(uttterance_mask_tensor)

    # Stack into final batch
    input_ids = torch.stack(padded_input_ids)
    attention_mask = torch.stack(padded_attention_mask) # [B, T, L]
    labels = torch.stack(padded_labels)
    utterance_mask = torch.stack(padded_utterance_mask) # Utterance is different from attention, size [B, T]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        'utterance_mask': utterance_mask
    }

def build_conversation_dataloader(data, batch_size, do_shuffling):
    dataset = EmotionDataset(data)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=do_shuffling,
        collate_fn=lambda batch: conversation_collate_fn(batch) # Can be expanded to other tokenizer if needed
    )

    return dataloader