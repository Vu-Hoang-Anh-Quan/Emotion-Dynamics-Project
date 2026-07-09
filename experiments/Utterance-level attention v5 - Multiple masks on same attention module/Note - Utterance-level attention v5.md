## Process
### Dataset: DailyDialog
Produce conversation like datasets
```python
{
    "conversation": [
        {"text": str, "speaker": int},
        {"text": str, "speaker": int},
        ...
    ]
}
```

### Pipeline
DailyDialog
→ flatten conversations
→ (context, current_text, speaker, emotion)

Build Dataset + DataLoader
→ custom collate_fn for batching

Tokenization
→ BERT tokenizer
→ truncate / pad (with context concatenation if used)

### Single-utterance classification
- BERT encoder: Encode each utterance with mean pooling then project to 256
- Linear layer: 256 -> 7: linear layer map

### Conversation classification
- BERT encoder: Encode each utterance with mean pooling then project to 256
- Transformer:
    - 8 heads self attention, each having 4 types of masks -> creating 4 outputs -> concatenated -> Linear into 768
    - Feed Forward Network with middle size of 3072 then back to 768
    - Stack again for 4 layers
- Classifier head:
→ Linear (7)
→ logits

### Training
Fine-tune the single-utterance classification, specifically BERT first, before concatenate it into the main pipeline.

## Hypothesis
Different masking might enables transformer to learns different interaction patterns that can enrich understanding.

## Run result
Single-utterance pipeline: Val 9.3488 F1-m, Test 0.4363 F1-m

Setup: 5 epochs for embedding fine-tuning, 30 epochs for whole architecture fine-tuning, dropout rate 0.3, lr 5e-5, batch size 4
- Best result: Val 0.3570 F1-m, Test 0.4463 F1-m

Setup: 5 epochs for embedding fine-tuning, 30 epochs for whole architecture fine-tuning, dropout rate 0.3, lr 5e-5, batch size 4, increase number of attention layers to 8
- Best result: Val 0.3716 F1-m, Test 0.4384 F1-m

Setup: 5 epochs for embedding fine-tuning, 30 epochs for whole architecture fine-tuning, dropout rate 0.3, lr 5e-5, batch size 16, increase number of attention layers to 8
- Best result: Val 0.3406 F1-m, Test 0.4410 F1-m

## Inference
Compare to vanilla transformer, the result is almost identical over experiments. 