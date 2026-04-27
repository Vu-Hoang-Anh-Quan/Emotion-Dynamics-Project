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
Epoch 1/5
100% 5449/5449 [11:40<00:00,  7.78it/s]
Train Loss: 0.3788
Val Loss:   0.2461 | Val Acc: 0.9114 | Val F1-score macro: 0.3958 | Val F1-score macro non-Neutral: 0.3030
Current model saved to directory /content/drive/MyDrive/Emotional Dynamics Project/saved_models/Baseline v3 - Context processing.pt

Epoch 2/5
100% 5449/5449 [11:19<00:00,  8.02it/s]
Train Loss: 0.2884
Val Loss:   0.2504 | Val Acc: 0.9067 | Val F1-score macro: 0.3992 | Val F1-score macro non-Neutral: 0.3076
Current model saved to directory /content/drive/MyDrive/Emotional Dynamics Project/saved_models/Baseline v3 - Context processing.pt

Epoch 3/5
100% 5449/5449 [11:20<00:00,  8.01it/s]
Train Loss: 0.2266
Val Loss:   0.2825 | Val Acc: 0.9002 | Val F1-score macro: 0.4478 | Val F1-score macro non-Neutral: 0.3649
Current model saved to directory /content/drive/MyDrive/Emotional Dynamics Project/saved_models/Baseline v3 - Context processing.pt

Epoch 4/5
100% 5449/5449 [11:19<00:00,  8.02it/s]
Train Loss: 0.1848
Val Loss:   0.3496 | Val Acc: 0.9032 | Val F1-score macro: 0.4143 | Val F1-score macro non-Neutral: 0.3255

Epoch 5/5
100% 5449/5449 [11:20<00:00,  8.01it/s]
Train Loss: 0.1564
Val Loss:   0.4176 | Val Acc: 0.8907 | Val F1-score macro: 0.3892 | Val F1-score macro non-Neutral: 0.2974
Final test loss: 0.6687
Final test accuracy: 0.8439
Final F1-score macro: 0.4785
Final F1-score macro non-Neutral: 0.4070
'''

## Inference
It can be seen that the model started overfitting from epoch 4 onwards.
The score F1-score between val and test datasets now only vary at around 0.04, which is much lower than previous versions. This confirms the hypothesis of stability.
However, score do not improve greatly.
