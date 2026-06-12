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
Hypothesis: Through concatenating x with output from attention, we expect the model to perform better, as more utterance useful information were transfered to the classifier. Therefore, the model should be more stable or better than the previous version by a small margin at least.

## Run result
Class 0: 78.6563
Class 1: 1.1111
Class 2: 0.4005
Class 3: 0.0000
Class 4: 15.6589
Class 5: 1.6667
Class 6: 2.5065
- Best val result: 0.0992
- Test result: 0.1116

## Inference
It seems like the model collapsed to learn the least common class, with result kinda the same to v4. However, v4 is acutally better as it can predict all classes. 

However, the problem might stems from the projection from BERT. Even though the embedding model achieved result almost identical to before, 768 dimensions compared to 256 is another clear bottleneck. 
