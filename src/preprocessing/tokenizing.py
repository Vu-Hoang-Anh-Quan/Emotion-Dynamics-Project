from transformers import AutoTokenizer
import os
import torch

tokenizer: AutoTokenizer

def load_tokenizer():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def save_tokenized_data(data, save_path, max_len=128):
    global tokenizer

    tokenized = []

    for item in data:
        text = f"[S{item['speaker']}] {item['text']}"

        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )

        tokenized.append({
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "emotion": torch.tensor(item["emotion"])
        })

    # ensure folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(tokenized, save_path)
    print(f"Saved to {save_path}")