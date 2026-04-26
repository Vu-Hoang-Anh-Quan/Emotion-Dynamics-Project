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
Epoch 1/5
  0% 0/5449 [00:00<?, ?it/s]W0426 09:26:17.326000 2648 torch/_inductor/utils.py:1731] [0/0] Not enough SMs to use max_autotune_gemm mode
100% 5449/5449 [09:55<00:00,  9.15it/s]
Train Loss: 0.4169
Val Loss:   0.2533 | Val Acc: 0.9057 | Val F1-score macro: 0.2403
Current model saved to directory /content/drive/MyDrive/Emotional Dynamics Project/saved_models/baseline_model_using_bert.pt

Epoch 2/5
100% 5449/5449 [07:21<00:00, 12.34it/s]
Train Loss: 0.3409
Val Loss:   0.2521 | Val Acc: 0.9067 | Val F1-score macro: 0.2624
Current model saved to directory /content/drive/MyDrive/Emotional Dynamics Project/saved_models/baseline_model_using_bert.pt

Epoch 3/5
100% 5449/5449 [07:23<00:00, 12.28it/s]
Train Loss: 0.2883
Val Loss:   0.2833 | Val Acc: 0.8966 | Val F1-score macro: 0.3042

Epoch 4/5
100% 5449/5449 [07:19<00:00, 12.39it/s]
Train Loss: 0.2426
Val Loss:   0.3295 | Val Acc: 0.8865 | Val F1-score macro: 0.3317

Epoch 5/5
100% 5449/5449 [07:19<00:00, 12.40it/s]
Train Loss: 0.2079
Val Loss:   0.3773 | Val Acc: 0.8923 | Val F1-score macro: 0.3253
Final test loss: 0.6127
Final test accuracy: 0.8393
Final F1-score macro: 0.4022
'''

## Inference
Final F1-score macro excluding Neutral has raised to 0.4, which indicates that the model is now capable of predicting rare emotion classes. 

We can see that even though the train loss gradually decreased over epochs, val loss started to increase from epoch 3 onwards, indicating that the model in overfitting. Moreover, the F1-score macro rises consistenly with that, also showing that this score might be very misleading is being used as benchmark for early stopping in training.

However, the cause might comes from high dropout rate (0.3), which makes the model not stable enough. It can also be the shallow classifying head that unable to improve the way model classify emotions.

Yet, it's clear that learning_rate of 1e-5 makes the model really learn, compared to 1e-3. 