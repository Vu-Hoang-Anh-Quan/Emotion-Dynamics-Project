## Process
### Dataset: DailyDialog
Separate each conversation into training samples of (context → current utterance):

```python
{
    "context": [
        {"text": str, "speaker": int},  # (i-1), most recent
        {"text": str, "speaker": int},  # (i-2)
        ...
    ],
    "current": {
        "text": str,        # target utterance
        "speaker": int,     # 0 / 1
        "emotion": int      # label (0–6)
    }
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
- BERT encoder: Encode each utterance with mean pooling
- 768 -> 7: linear layer map

### Conversation classification
- BERT encoder: Encode each utterance with mean pooling
- Self-attention layer + residual: 128
    - Casual mask to ensure temporal order
- Classifier head:
→ Linear (128)
→ LayerNorm
→ ReLU
→ Dropout
→ Linear (7)
→ logits

### Training
Fine-tune the single-utterance classification, specifically BERT first, before freezing and concatenating into the main pipeline.

## Hypothesis

## Run result
```
```

## Inference
The final result is actually the same as v1. This shows that the attention is actually harmful.

Next experiments: make residual really residual and also inspect attention weights if they are washing out signals.