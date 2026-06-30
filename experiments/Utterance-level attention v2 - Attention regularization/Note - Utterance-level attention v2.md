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
- BERT encoder: Encode each utterance with mean pooling then project to 256
- Linear layer: 256 -> 7: linear layer map

### Conversation classification
- BERT encoder: Encode each utterance with mean pooling then project to 256
- Self-attention layer + residual: 
    - Casual mask to ensure temporal order -> 256
    - Residual: concat: x | output
- Classifier head:
→ Linear (128)
→ LayerNorm
→ ReLU
→ Dropout
→ Linear (7)
→ logits

### Training
Fine-tune the single-utterance classification, specifically BERT first, before concatenate it into the main pipeline.
Add 0.1-0.2 regularization loss for attention probability, using renyi entropy of 2.

## Hypothesis
Hypothesis: by rewarding the attention probability to focus on exactly 2 utterances, we hope the layer to capture the most important utterances, usually is 2 (including itself) instead of averaging everything. Therefore, the score should at least improve compared to separated-training versions

## Run result
Setup: 5 epoch for embedding fine-tuning, 10 epochs for whole architecture fine-tuning
- Best epoch result:
Epoch 8/10
| Class 0: 75.0651
| Class 1: 1.0286
| Class 2: 0.1735
| Class 3: 0.2355
| Class 4: 19.2713
| Class 5: 1.7350
| Class 6: 2.4910
Train Loss: 1.6336
Val Loss: 1.5590 | Val Acc: 0.7230 | Val F1-score macro: 0.2057 | Val F1-score macro non-Neutral: 0.1005
- Final result:
Class 0: 67.0543
Class 1: 1.3307
Class 2: 1.0078
Class 3: 0.2326
Class 4: 23.9535
Class 5: 2.9070
Class 6: 3.5142

Test Loss: 1.1294
Test Acc: 0.6297
Test F1-score macro: 0.2185
Test F1-score macro non-Neutral: 0.1269

## Inference
The problem is the collapse of the whole module, not just the simple probability of attention. This suggest that the specific single layer of self-attention is actually harmful.

Next work: using transformer