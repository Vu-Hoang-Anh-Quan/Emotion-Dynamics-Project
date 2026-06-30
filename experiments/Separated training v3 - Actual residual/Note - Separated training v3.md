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
The final result is is actually worse compared to v2. It's clear that the attention mechanism itself is the problem.
Moreover, take a look, we can see that the model always failed to learn rare emotion classes, while class 1 and 4 just dominant. The attention mechanism is actually weighted averaging of utterances, that's why most of the time, it just diverge towards the common emotion in the conversation. After inspecting attention weights, it is clear that the attention actually weight the current utterance at only around 0.35 at most, which diluted the useful information from BERT. Sometimes, the probability went up to 0.5, which means the attention mechanism is actually learning. However, the problem was that the model cannot risky to take a big step towards prioritizing the current utterance too much, compared ot others. 

Next work: try to preserve current utterance information. 

Update: the bug actually stems from load_state_dict. Due to mistakes in wrapping the model name, the weight of the pretrained model cannot be loaded, thus making result identical to that of custom pooling v1.