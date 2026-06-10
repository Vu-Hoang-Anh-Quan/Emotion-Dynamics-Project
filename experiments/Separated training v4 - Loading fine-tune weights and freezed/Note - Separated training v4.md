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
- Self-attention layer + residual: 768
    - Casual mask to ensure temporal order
    - Residual: x + output
- Classifier head:
→ Linear (128)
→ LayerNorm
→ ReLU
→ Dropout
→ Linear (7)
→ logits

### Training
Fine-tune the single-utterance classification, specifically BERT first, before concatenate it into the main pipeline

## Hypothesis
Hypothesis: Through separated training, we hope that BERT will better capture information from each utterance, therefore provide much more semantics when combine with context during full training, improving the final score, at least by a margin compared to baseline v4. 
However, the problem about self-attention averaging everything altogether has not been fixed. Therefore, the model would likely outperform custom pooling v1, but cannot get over baseline v4.

## Run result
```
```

## Inference
Next work: try to preserve current utterance information. 