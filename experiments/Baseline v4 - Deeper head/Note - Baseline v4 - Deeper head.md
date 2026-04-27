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
- Extract CLS -> CLS (768) -> hidden layer (256) -> LayerNorm -> ReLU -> logits (7)
- CrossEntropyLoss

## Changes compared to Baseline v2
Making the head deeper with another layer and layer norm

## Hypothesis
This will certainly makes the model able to recognize more complex samples, especially with the adding of context that it needs to process much more information. However, it can also more prone to overfitting.

## Run result
'''
Epoch 1/5
  0% 0/5449 [00:00<?, ?it/s]W0427 11:32:57.100000 2678 torch/_inductor/utils.py:1731] [0/0] Not enough SMs to use max_autotune_gemm mode
100% 5449/5449 [14:49<00:00,  6.12it/s]
Train Loss: 0.3959
Val Loss:   0.2531 | Val Acc: 0.9063 | Val F1-score macro: 0.3708 | Val F1-score macro non-Neutral: 0.2744
Current model saved

Epoch 2/5
100% 5449/5449 [11:23<00:00,  7.97it/s]
Train Loss: 0.2988
Val Loss:   0.2522 | Val Acc: 0.9135 | Val F1-score macro: 0.3752 | Val F1-score macro non-Neutral: 0.2789
Current model saved

Epoch 3/5
100% 5449/5449 [11:23<00:00,  7.97it/s]
Train Loss: 0.2394
Val Loss:   0.2939 | Val Acc: 0.9067 | Val F1-score macro: 0.4360 | Val F1-score macro non-Neutral: 0.3504
Current model saved

Epoch 4/5
100% 5449/5449 [11:24<00:00,  7.96it/s]
Train Loss: 0.1962
Val Loss:   0.3301 | Val Acc: 0.8969 | Val F1-score macro: 0.4638 | Val F1-score macro non-Neutral: 0.3838
Current model saved

Epoch 5/5
100% 5449/5449 [11:23<00:00,  7.97it/s]
Train Loss: 0.1657
Val Loss:   0.3964 | Val Acc: 0.8877 | Val F1-score macro: 0.4691 | Val F1-score macro non-Neutral: 0.3910
Current model saved
Final test loss: 0.6114
Final test accuracy: 0.8371
Final F1-score macro: 0.4988
Final F1-score macro non-Neutral: 0.4314
'''

## Inference
It can be seen that the model started overfitting from epoch 4 onwards.
The score F1-score between val and test datasets now only vary at around 0.04, which is much lower than previous versions. This confirms the hypothesis of stability.
However, score do not improve greatly.
