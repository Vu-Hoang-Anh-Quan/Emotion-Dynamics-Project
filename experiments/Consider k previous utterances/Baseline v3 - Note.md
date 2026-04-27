# Baseline with dynamic padding experiment note

# Baseline experiment note

## Config
'''
{
  "batch_size": 16,
  "dataset_name": "daily_dialog",
  "debug": 0,
  "dropout_rate": 0.3,
  "embedding_model_name": "bert-base-uncased",
  "epochs": 5,
  "experiment_name": "Fine-tuning with lower lr",
  "learning_rate": 1e-05,
  "need_to_retrain": 1,
  "num_labels": 7,
  "prepare_data_again": 0,
  "resulting_model_name": "baseline_model_using_bert",
  "seed": 42,
  "use_cuda": 1
}
'''

## Process
Using dataset "DailyDialog"

Separate each conversations into multiple utterances, each have:
'''
{
    "context": [
        {"text": str, "speaker": int},  # Most recent (i-1)
        {"text": str, "speaker": int},  # (i-2)
        ...                             # Up to k-steps back
    ],
    "current": {
        "text": str,                    # The utterance to be classified
        "speaker": int,                 # 0 or 1
        "emotion": int                  # Ground truth label
    }
}
'''

- DailyDialog → flatten → (context(text, speaker), speaker, text, emotion)
- Tokenizer and Embeddings with BERT
- Extract CLS -> pass forward to logits (7 dims)
- CrossEntropyLoss

## Changes compared to Baseline v2
Including previous utterances in context representation (currently 3)

## Hypothesis
Through including k (currently 3) previous utterances, the model's performance will be much more stabilized across val and test dataset, and performance will increase as the model has more information to decide from.

## Run result
'''
'''

## Inference
