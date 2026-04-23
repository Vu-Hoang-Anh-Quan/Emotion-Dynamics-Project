from transformers import AutoTokenizer
import os
import torch

tokenizer: AutoTokenizer

def load_tokenizer():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def save_tokenized_data(data, save_path, maxlen=512):
    global tokenizer

    tokenized = []

    for item in data:
        text = f"[S{item['speaker']}] {item['text']}"

        encoding = tokenizer(
            text,
            truncation=True, # True with maxlen if needed
            max_length=maxlen,
            padding=False,
            return_tensors=None # Return Python lists, not tensors
        )

        tokenized.append({
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "emotion": torch.tensor(item["emotion"])
        })

    # ensure folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(tokenized, save_path)
    print(f"Saved to {save_path}")