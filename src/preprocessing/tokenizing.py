from transformers import AutoTokenizer
import os
import torch

tokenizer: AutoTokenizer

def load_tokenizer():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def build_context(context_turns):
    return " ".join([f"S{speaker}: {text}\n" for speaker, text in context_turns])

def save_tokenized_data(data, save_path, maxlen=512):
    global tokenizer

    tokenized = []

    for item in data:
        context_turns = item["context"]
        current_raw = item["current"]

        context = build_context(context_turns)
        current = f"S{current_raw['speaker']}: {current_raw['text']}"

        encoding = tokenizer(
            context,
            current,
            truncation="only_first", # True with maxlen if needed
            max_length=maxlen,
            padding=False,
            return_tensors=None # Return Python lists, not tensors
        )

        tokenized.append({
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "emotion": torch.tensor(current_raw["emotion"])
        })

    # ensure folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(tokenized, save_path)
    print(f"Saved to {save_path}")