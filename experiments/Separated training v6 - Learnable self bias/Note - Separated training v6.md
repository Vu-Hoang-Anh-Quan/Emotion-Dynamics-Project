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
Fine-tune the single-utterance classification, specifically BERT first, before concatenate it into the main pipeline

## Hypothesis
Hypothesis: Through concatenating x with output from attention, we expect the model to perform better, as more utterance useful information were transfered to the classifier. Therefore, the model should be more stable or better than the previous version by a small margin at least.

## Run result
### Test run
Setup: 1 epoch for embedding fine-tuning, 10 epochs for whole architecture fine-tuning
- Best epoch result:
Epoch 6/10
| Class 0: 81.9804
| Class 1: 0.9171
| Class 2: 0.1239
| Class 3: 0.0000
| Class 4: 13.7440
| Class 5: 1.4376
| Class 6: 1.7970
Train Loss: 1.6560
Val Loss: 1.5128 | Val Acc: 0.7794 | Val F1-score macro: 0.2196 | Val F1-score macro non-Neutral: 0.1103
- Final result:
Class 0: 75.2842
Class 1: 1.3824
Class 2: 0.2584
Class 3: 0.0000
Class 4: 17.4031
Class 5: 2.3773
Class 6: 3.2946

Test Loss: 1.0153
Test Acc: 0.6831
Test F1-score macro: 0.2098
Test F1-score macro non-Neutral: 0.1096

### Second run
Setup: 5 epoch for embedding fine-tuning, 10 epochs for whole architecture fine-tuning

- Best Val result
22:16 | loops | Epoch 2/10
22:17 | metrics | Class 0: 80.7783
22:17 | metrics | Class 1: 1.1402
22:17 | metrics | Class 2: 0.0000
22:17 | metrics | Class 3: 0.0000
22:17 | metrics | Class 4: 15.6525
22:17 | metrics | Class 5: 0.9667
22:17 | metrics | Class 6: 1.4624
22:17 | loops | Train Loss: 1.6488
22:17 | loops | Val Loss: 1.5563 | Val Acc: 0.7706 | Val F1-score macro: 0.2187 | Val F1-score macro non-Neutral: 0.1104

- Test result
22:25 | metrics | Class 0: 72.4677
22:25 | metrics | Class 1: 1.3566
22:25 | metrics | Class 2: 0.3618
22:25 | metrics | Class 3: 0.0258
22:25 | metrics | Class 4: 20.9432
22:25 | metrics | Class 5: 2.2351
22:25 | metrics | Class 6: 2.6098
22:25 | main | Conversation Pipeline:
Test Loss: 1.0155
Test Acc: 0.6678
Test F1-score macro: 0.2184
Test F1-score macro non-Neutral: 0.1220

### Third run
- Best val result
13:40 | loops | Epoch 1/10
13:40 | loops | Freeze BERT
13:41 | metrics | Class 0: 82.7488
13:41 | metrics | Class 1: 1.1154
13:41 | metrics | Class 2: 0.0000
13:41 | metrics | Class 3: 0.0000
13:41 | metrics | Class 4: 12.9260
13:41 | metrics | Class 5: 1.3632
13:41 | metrics | Class 6: 1.8466
13:41 | loops | Train Loss: 1.6835
13:41 | loops | Val Loss: 1.5262 | Val Acc: 0.7856 | Val F1-score macro: 0.2180 | Val F1-score macro non-Neutral: 0.1077

- Test result
13:51 | metrics | Class 0: 77.1189
13:51 | metrics | Class 1: 1.7829
13:51 | metrics | Class 2: 0.1680
13:51 | metrics | Class 3: 0.0000
13:51 | metrics | Class 4: 15.7752
13:51 | metrics | Class 5: 2.4935
13:51 | metrics | Class 6: 2.6615
13:51 | main | Conversation Pipeline:
Test Loss: 0.9704
Test Acc: 0.6961
Test F1-score macro: 0.2165
Test F1-score macro non-Neutral: 0.1158

## Inference
The model relied almost completely on the self bias to improve its score, which is why through training epochs, the score do not go up.
Until now, the attention architecture seems to not perform effectively, but rather harmfully to the whole task. The attention mechanism actually learn what it needs to, but it does not have the resolve to move to a complete bias of 0.9 probability over one utterance. Especially during rare emotion classes, the probability should focus entirely on 1 to 3 utterances, not everything equally, which is not happening with the current attention module.

Next work: try graph-based architecture
