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
    - 8 heads self attention
    - Feed Forward Network with middle size of 3072 then back to 768
    - Stack again for 4 layers
- Classifier head:
→ Linear (7)
→ logits

### Training
Fine-tune the single-utterance classification, specifically BERT first, before concatenate it into the main pipeline.

## Hypothesis
Applying both multi-head attention and stacking layers of attention and feed forward can produce better result in classification, compared to just a single-head single-layer self-attention.

## Run result
Setup: 5 epochs for embedding fine-tuning, 30 epochs for whole architecture fine-tuning, dropout rate 0.2 and 0.3, lr 1e-4 and 5e-5 (basically 4 experiments), batch size 4
- Best result: ~0.12 F1-m excluding Neutral

Setup: the same but with lr 1e-4, dropout 0.2, batch size 2
- Best result: Val F1-m 0.1987, Test F1-m 0.2484

## Inference
