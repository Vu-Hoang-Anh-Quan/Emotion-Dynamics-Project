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

### Second run
```json
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
  "experiment_name": "Sequential Modelling v2 - Weighted loss",
  "freeze_except_last_k": 2,
  "lr_bert": 1e-05,
  "lr_head": 0.001,
  "need_to_retrain": 1,
  "num_labels": 7,
  "prepare_data_again": 0,
  "resulting_model_name": "Sequential Modelling v2",
  "seed": 42,
  "use_cuda": 1,
  "weight_decay": 0.01
}
```
```
Epoch 1/6
100% 2780/2780 [02:25<00:00, 19.16it/s]
Class 0: 74.0116
Class 1: 0.0000
Class 2: 0.0000
Class 3: 0.0000
Class 4: 20.6965
Class 5: 0.3470
Class 6: 4.9449
Train Loss: 1.7430
Val Loss:   1.5827 | Val Acc: 0.7121 | Val F1-score macro: 0.1705 | Val F1-score macro non-Neutral: 0.0605
Current model saved to directory /content/drive/MyDrive/Emotional Dynamics Project/saved_models/Sequential Modelling v2.pt

Epoch 2/6
100% 2780/2780 [02:33<00:00, 18.07it/s]
Class 0: 82.4761
Class 1: 1.8094
Class 2: 0.0000
Class 3: 0.0000
Class 4: 12.6782
Class 5: 2.0201
Class 6: 1.0162
Train Loss: 1.6858
Val Loss:   1.5486 | Val Acc: 0.7823 | Val F1-score macro: 0.2073 | Val F1-score macro non-Neutral: 0.0955
Current model saved to directory /content/drive/MyDrive/Emotional Dynamics Project/saved_models/Sequential Modelling v2.pt

Epoch 3/6
100% 2780/2780 [02:36<00:00, 17.71it/s]
Class 0: 90.7423
Class 1: 1.2517
Class 2: 0.0000
Class 3: 0.0000
Class 4: 5.9363
Class 5: 0.9791
Class 6: 1.0906
Train Loss: 1.6732
Val Loss:   1.5252 | Val Acc: 0.8369 | Val F1-score macro: 0.2076 | Val F1-score macro non-Neutral: 0.0903

Epoch 4/6
100% 2780/2780 [02:38<00:00, 17.59it/s]
Class 0: 76.4779
Class 1: 0.4462
Class 2: 0.1239
Class 3: 0.0000
Class 4: 19.0358
Class 5: 2.0077
Class 6: 1.9085
Train Loss: 1.6768
Val Loss:   1.5212 | Val Acc: 0.7326 | Val F1-score macro: 0.1958 | Val F1-score macro non-Neutral: 0.0879

Epoch 5/6
100% 2780/2780 [02:37<00:00, 17.70it/s]
Class 0: 77.7668
Class 1: 0.8303
Class 2: 0.0000
Class 3: 0.0000
Class 4: 18.3666
Class 5: 1.5739
Class 6: 1.4624
Train Loss: 1.6698
Val Loss:   1.5161 | Val Acc: 0.7427 | Val F1-score macro: 0.1946 | Val F1-score macro non-Neutral: 0.0853

Epoch 6/6
100% 2780/2780 [02:37<00:00, 17.60it/s]
Class 0: 78.4360
Class 1: 1.7350
Class 2: 0.0000
Class 3: 0.0000
Class 4: 15.5038
Class 5: 1.6607
Class 6: 2.6645
Train Loss: 1.6625
Val Loss:   1.5063 | Val Acc: 0.7484 | Val F1-score macro: 0.1939 | Val F1-score macro non-Neutral: 0.0837

Class 0: 73.8114
Class 1: 2.9716
Class 2: 0.0000
Class 3: 0.0000
Class 4: 15.8915
Class 5: 3.0103
Class 6: 4.3152
Final test loss: 1.1611
Final test accuracy: 0.6685
Final F1-score macro: 0.1901
Final F1-score macro non-Neutral: 0.0879
```

## Inference
The model cleary underperforms, with terrible classification of minor classes, even after trying its best. However, it is actually learning and achieve something, compared to the previous version.