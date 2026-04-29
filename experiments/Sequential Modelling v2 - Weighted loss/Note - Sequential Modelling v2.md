# Baseline with dynamic padding experiment note

# Baseline experiment note

## Config
```json
{
  "batch_size": 16,
  "consider_previous_k": 3,
  "dataset_name": "daily_dialog",
  "debug": 0,
  "deterministic_run": 0,
  "dropout_bert": 0.1,
  "dropout_head": 0.3,
  "embedding_model_name": "bert-base-uncased",
  "epochs": 6,
  "experiment_name": "Sequential Modelling v1",
  "freeze_except_last_k": 4,
  "lr_bert": 1e-05,
  "lr_head": 0.0005,
  "need_to_retrain": 1,
  "num_labels": 7,
  "prepare_data_again": 0,
  "resulting_model_name": "Sequential Modelling v1",
  "seed": 42,
  "use_cuda": 1,
  "weight_decay": 0.01
}
```

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

## Changes compared to Sequential Modelling v1
Add weighted loss

## Hypothesis
By having weighted loss, the model will actually learning instead of collapsing to the easy Neutral way

## Run result
### First run
```
Epoch 1/6
  0% 0/695 [00:00<?, ?it/s]W0429 14:43:58.576000 2088 torch/_inductor/utils.py:1731] [0/0_1] Not enough SMs to use max_autotune_gemm mode
100% 695/695 [06:47<00:00,  1.71it/s]
Class 0: 12.3931
Class 1: 0.0000
Class 2: 0.0000
Class 3: 0.0000
Class 4: 66.8484
Class 5: 0.0000
Class 6: 20.7585
Train Loss: 1.9453
Val Loss:   1.8222 | Val Acc: 0.1864 | Val F1-score macro: 0.0622 | Val F1-score macro non-Neutral: 0.0347
Current model saved to directory /content/drive/MyDrive/Emotional Dynamics Project/saved_models/Sequential Modelling v2.pt

Epoch 2/6
100% 695/695 [04:22<00:00,  2.64it/s]
Class 0: 10.6085
Class 1: 1.7846
Class 2: 0.0000
Class 3: 0.0000
Class 4: 48.7173
Class 5: 8.7247
Class 6: 30.1648
Train Loss: 1.9203
Val Loss:   1.8280 | Val Acc: 0.1612 | Val F1-score macro: 0.0657 | Val F1-score macro non-Neutral: 0.0437
Current model saved to directory /content/drive/MyDrive/Emotional Dynamics Project/saved_models/Sequential Modelling v2.pt
```

## Inference
