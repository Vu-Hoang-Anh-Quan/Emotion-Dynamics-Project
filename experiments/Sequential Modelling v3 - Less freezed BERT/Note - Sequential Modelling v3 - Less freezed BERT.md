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
- GRU: Process by timestep
- Classifier head:
rep (768)
→ Linear (256)
→ LayerNorm
→ ReLU
→ Dropout
→ Linear (7)
→ logits

## Changes compared to Sequential Modelling v2
Unfreeze 8 layers of BERT, instead of 4-2.

## Hypothesis
Hypothesis: allow the model to get more information after going through BERT and less depend on GRU, thus increase efficiency.

## Run result
```
{
  "batch_size": 4,
  "compile_model": 0,
  "consider_previous_k": 3,
  "dataset_name": "daily_dialog",
  "debug": 0,
  "deterministic_run": 0,
  "dropout_bert": 0.1,
  "dropout_head": 0.3,
  "embedding_model_name": "bert-base-uncased",
  "epochs": 6,
  "experiment_name": "Sequential Modelling v3 - Less freezed BERT",
  "freeze_except_last_k": 8,
  "lr_bert": 1e-05,
  "lr_head": 0.001,
  "need_to_retrain": 1,
  "num_labels": 7,
  "prepare_data_again": 0,
  "resulting_model_name": "Sequential Modelling v3",
  "seed": 42,
  "use_cuda": 1,
  "weight_decay": 0.01
}
Epoch 1/6
Train Loss: 1.7305
Val Loss:   1.5880 | Val Acc: 0.6897 | Val F1-score macro: 0.1704 | Val F1-score macro non-Neutral: 0.0635
Current model saved to directory /home/Quan/Desktop/Emotion-Dynamics-Project/saved_models/Sequential Modelling v3.pt
Epoch 2/6
Train Loss: 1.6933
Val Loss:   1.5372 | Val Acc: 0.7909 | Val F1-score macro: 0.1926 | Val F1-score macro non-Neutral: 0.0774
Current model saved to directory /home/Quan/Desktop/Emotion-Dynamics-Project/saved_models/Sequential Modelling v3.pt
Epoch 3/6
Train Loss: 1.6793
Val Loss:   1.5399 | Val Acc: 0.8296 | Val F1-score macro: 0.2013 | Val F1-score macro non-Neutral: 0.0836
Current model saved to directory /home/Quan/Desktop/Emotion-Dynamics-Project/saved_models/Sequential Modelling v3.pt
Epoch 4/6
Train Loss: 1.6675
Val Loss:   1.5237 | Val Acc: 0.7609 | Val F1-score macro: 0.1970 | Val F1-score macro non-Neutral: 0.0861
Current model saved to directory /home/Quan/Desktop/Emotion-Dynamics-Project/saved_models/Sequential Modelling v3.pt
Epoch 5/6
Train Loss: 1.6599
Val Loss:   1.5306 | Val Acc: 0.7420 | Val F1-score macro: 0.1952 | Val F1-score macro non-Neutral: 0.0863
Current model saved to directory /home/Quan/Desktop/Emotion-Dynamics-Project/saved_models/Sequential Modelling v3.pt
Epoch 6/6
Train Loss: 1.6563
Val Loss:   1.5252 | Val Acc: 0.7091 | Val F1-score macro: 0.1879 | Val F1-score macro non-Neutral: 0.0813

Class 0: 70.5297
Class 1: 3.5271
Class 2: 0.0000
Class 3: 0.0000
Class 4: 20.8398
Class 5: 2.1964
Class 6: 2.9070
Final test loss: 1.1617
Final test accuracy: 0.6508
Final F1-score macro: 0.1903
Final F1-score macro non-Neutral: 0.0908
```

## Inference
Compare to the last version, the F1-m non-Neutral increased by a small margin: 0.01. This actually shows that the model moved a little bit closer to predicting hard classes instead of just guessing Neutral, but is still far from achieving the Baseline.

Actually, there are two points of information loss:
1. At GRU, where important but weak signals evaporate
2. At the CLS, where BERT just collapse everything into a next-utterance predicting task instead of identifying emotion. Even though this work in the baseline, but the baseline is basically single CLS classification. Therefore, weak signals there can also be looked at. But when combine with context processing, the signals are weakened rapidly.

Next work: change the pipeline to account for the loss of information detected in SM v2 experiment