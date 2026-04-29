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

## Changes compared to Baseline v4
Instead of manually concatenate context, the model use GRU to handels temporal relations

## Hypothesis
By considering much more context, the model will likely to perform better than Baseline v4, at least by a small margin.

## Run result
```
Epoch 1/6
100% 695/695 [04:54<00:00,  2.36it/s]
Class 0: 100.0000
Class 1: 0.0000
Class 2: 0.0000
Class 3: 0.0000
Class 4: 0.0000
Class 5: 0.0000
Class 6: 0.0000
Train Loss: 0.6360
Val Loss:   0.4913 | Val Acc: 0.8809 | Val F1-score macro: 0.1338 | Val F1-score macro non-Neutral: 0.0000
```

## Inference
With a 100% prediction on class 0 - Neutral, we can clearly see that the model collapsed entirely with the dataset. 
Proposing solutions: Weighted loss