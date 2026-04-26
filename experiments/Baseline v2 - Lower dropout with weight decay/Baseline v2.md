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
{
    'text': the line of talk
    'emotion' (int): a lable of emotion
    'speaker' (int): 0-1, first or second speaker
}

DailyDialog → flatten → (text, speaker, emotion)
           → tokenizer → embeddings
           → model (RNN / Transformer / etc.)
           → linear layer → logits (7-dim)
           → CrossEntropyLoss

## Changes compared to baseline
Change learning_rate from 1e-3 to 1e-5

## Run result
'''
Train Loss: 0.4208
Val Loss:   0.2608 | Val Acc: 0.9021 | Val F1-score macro: 0.3405 | Val F1-score macro non-Neutral: 0.2395
Current model saved

Train Loss: 0.3415
Val Loss:   0.2588 | Val Acc: 0.9056 | Val F1-score macro: 0.3894 | Val F1-score macro non-Neutral: 0.2961
Current model saved

Train Loss: 0.2906
Val Loss:   0.3011 | Val Acc: 0.8870 | Val F1-score macro: 0.4175 | Val F1-score macro non-Neutral: 0.3309

Final test loss: 0.4049
Final test accuracy: 0.8578
Final F1-score macro: 0.4434
Test F1-score macro non-Neutral: 0.3643
'''

## Inference
It seems that the model's F1-m non-Neutral score only stays at 0.36, lower than previous runtime, even though achieving higher on val dataset.

Moreover, due to not processing context, the f1-score between val and test data varies greatly. This indicates that the result might not be that reliable when consider model's efficiency. 