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
```

## Inference
The F1-m outperforms all previous versions on both Val and Test dataset, thus confirming the hypothesis of better classification. Moroever, this is the first time that F1-score still increases after epoch 3, showing how deeper head creates more possibility for better model performance. 