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

## Hypothesis
Hypothesis: Changing the pooling from CLS to mean pooling should leave the performance at least the same as CLS, or somewhat a little worse. This might not be the bottleneck yet, but worth a try. 

## Run result
```
Epoch 1/8
Class 0: 85.7479
Class 1: 0.0000
Class 2: 0.0000
Class 3: 0.0000
Class 4: 11.8726
Class 5: 0.0991
Class 6: 2.2803
Train Loss: 1.7215
Val Loss:   1.6332 | Val Acc: 0.8025 | Val F1-score macro: 0.1800 | Val F1-score macro non-Neutral: 0.0616
Current model saved

Epoch 2/8
Class 0: 81.9308
Class 1: 1.2021
Class 2: 0.0000
Class 3: 0.0000
Class 4: 16.2102
Class 5: 0.0000
Class 6: 0.6568
Train Loss: 1.7115
Val Loss:   1.5512 | Val Acc: 0.7751 | Val F1-score macro: 0.1948 | Val F1-score macro non-Neutral: 0.0820
Current model saved

Epoch 3/8
Class 0: 87.8176
Class 1: 0.6197
Class 2: 0.0000
Class 3: 0.0000
Class 4: 10.6085
Class 5: 0.0000
Class 6: 0.9543
Train Loss: 1.6820
Val Loss:   1.6094 | Val Acc: 0.8227 | Val F1-score macro: 0.2076 | Val F1-score macro non-Neutral: 0.0920
Current model saved

Epoch 4/8
Class 0: 59.5613
Class 1: 0.8551
Class 2: 0.1611
Class 3: 0.0000
Class 4: 36.9934
Class 5: 1.2517
Class 6: 1.1773
Train Loss: 1.6853
Val Loss:   1.5448 | Val Acc: 0.5907 | Val F1-score macro: 0.1761 | Val F1-score macro non-Neutral: 0.0833

Epoch 5/8
Class 0: 88.0530
Class 1: 1.3508
Class 2: 0.0000
Class 3: 0.0000
Class 4: 6.9277
Class 5: 1.5987
Class 6: 2.0696
Train Loss: 1.6702
Val Loss:   1.5198 | Val Acc: 0.8196 | Val F1-score macro: 0.2129 | Val F1-score macro non-Neutral: 0.0980
Current model saved

Epoch 6/8
Class 0: 86.3428
Class 1: 0.6692
Class 2: 0.0496
Class 3: 0.0000
Class 4: 10.7820
Class 5: 0.5205
Class 6: 1.6359
Train Loss: 1.6739
Val Loss:   1.5662 | Val Acc: 0.8108 | Val F1-score macro: 0.2051 | Val F1-score macro non-Neutral: 0.0900

Epoch 7/8
Class 0: 78.5847
Class 1: 0.2974
Class 2: 0.2107
Class 3: 0.0000
Class 4: 19.5687
Class 5: 0.1735
Class 6: 1.1650
Train Loss: 1.6720
Val Loss:   1.5732 | Val Acc: 0.7477 | Val F1-score macro: 0.2115 | Val F1-score macro non-Neutral: 0.1045
Current model saved

Epoch 8/8
Class 0: 86.1073
Class 1: 0.8675
Class 2: 0.0372
Class 3: 0.0000
Class 4: 11.8230
Class 5: 0.0496
Class 6: 1.1154
Train Loss: 1.6590
Val Loss:   1.6160 | Val Acc: 0.8104 | Val F1-score macro: 0.2019 | Val F1-score macro non-Neutral: 0.0864

Class 0: 69.5220
Class 1: 0.3618
Class 2: 1.0465
Class 3: 0.0000
Class 4: 26.4729
Class 5: 0.8527
Class 6: 1.7442
Final test loss: 1.0862
Final test accuracy: 0.6490
Final F1-score macro: 0.1978
Final F1-score macro non-Neutral: 0.1005
```

## Inference
It seems that due to the nature of emotion being the combination of all emotional semantics among the utterance, mean pooling outperformed CLS on the same benchmask, reaching 0.1 instead of just 0.08~0.09. Moreover, it can be seen that the model was able to learn the second least common emotion - class 2. This shows that mean pooling and other custom pooling has the potential to outperform CLS.

However, the bottleneck is yet to be resolved, when score is still ridiculously small compared to baseline v2. The problem now is very likely to lie in the gradients flowing from loss, through attention to BERT is too far away, making it weak and noisy. 
Therefore, the next work will lies on separated training session, with single-utterance recognition part being trained first, before concatenated into the main pipeline.