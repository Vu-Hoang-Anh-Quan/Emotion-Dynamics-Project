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

### Model
- BERT encoder: Encode each utterance
- Self-attention layer + residual: 128
    - Casual mask to ensure temporal order
- Classifier head:
→ Linear (128)
→ LayerNorm
→ ReLU
→ Dropout
→ Linear (7)
→ logits

## Hypothesis
Hypothesis: Fixing the bottleneck of context processing from GRU, which only allows linear information transfer. Therefore, the result is expected to at least promising to rival that of GRU.

## Run result
```
Epoch 1/10
Train Loss: 1.7267
Val Loss:   1.6509 | Val Acc: 0.7996 | Val F1-score macro: 0.1926 | Val F1-score macro non-Neutral: 0.0767
Current model saved
Epoch 2/10
Train Loss: 1.7130
Val Loss:   1.5500 | Val Acc: 0.7482 | Val F1-score macro: 0.1888 | Val F1-score macro non-Neutral: 0.0780
Current model saved
Epoch 3/10
Train Loss: 1.6825
Val Loss:   1.6237 | Val Acc: 0.8167 | Val F1-score macro: 0.2003 | Val F1-score macro non-Neutral: 0.0840
Current model saved
Epoch 4/10
Train Loss: 1.6924
Val Loss:   1.5521 | Val Acc: 0.5503 | Val F1-score macro: 0.1707 | Val F1-score macro non-Neutral: 0.0834
Epoch 5/10
Train Loss: 1.6725
Val Loss:   1.5245 | Val Acc: 0.8390 | Val F1-score macro: 0.2102 | Val F1-score macro non-Neutral: 0.0931
Current model saved
Epoch 6/10
Train Loss: 1.6746
Val Loss:   1.5758 | Val Acc: 0.8132 | Val F1-score macro: 0.1976 | Val F1-score macro non-Neutral: 0.0811
Epoch 7/10
Train Loss: 1.6699
Val Loss:   1.5728 | Val Acc: 0.7764 | Val F1-score macro: 0.1907 | Val F1-score macro non-Neutral: 0.0771

Class 0: 89.1990
Class 1: 1.9251
Class 2: 0.0000
Class 3: 0.0000
Class 4: 5.5426
Class 5: 1.7313
Class 6: 1.6021
Final test loss: 0.9642
Final test accuracy: 0.7611
Final F1-score macro: 0.1894
Final F1-score macro non-Neutral: 0.0765
```

## Inference
Compare to GRU, the result is similar. However, not to judge too early, as the score is ridiculously small, which shows that the bottleneck has not been completely been fixed. Compare this to manual connection version - Baseline v3, it's clear that the score is terrible.

Therefore, the bottleneck of utterance-level processing approach is actually the vectorized representation of utterance. It seems that with only BERT CLS, the model will easily collapsed as CLS is not designed for and thus cannot capture important emotional semantics among utterances. This prompt the next approach: improve utterance representation.

Future work: Try fine-tune BERT first for single-utterance emotion recognition (training), and/or apply custom pooling in place of CLS.

However, such approach of fine-tune BERT first will gone back to baseline v2, which face the classic problem of test results do not consider context, which will hinder fine-tuned BERT if wanted to combine it with other complex context processing architecture. Therefore, the approach of baseline v3, consider k utterances before, is worth trying as experiment when fine-tuning BERT for single-utterance emotion recognition, as it's currently the best performance among all models so far.