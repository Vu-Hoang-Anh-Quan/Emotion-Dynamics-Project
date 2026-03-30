from transformers import AutoTokenizer

tokenizer: AutoTokenizer

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_data(data, tokenizer, max_len=128):
    tokenized = []

    for item in data:
        # Option A: inject speaker into text
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
            "emotion": item["emotion"]
        })

    return tokenized