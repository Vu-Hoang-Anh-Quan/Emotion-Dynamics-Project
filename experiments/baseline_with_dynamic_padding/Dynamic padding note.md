# Baseline with dynamic padding experiment note

# Baseline experiment note

## Config
'''
{
  "experiment_name": "baseline_with_dynamic_padding_and_gpu",
  "seed": 42,
  "dataset_name": "daily_dialog",
  "learning_rate": 0.001,
  "num_labels": 7,
  "dropout_rate": 0.3,
  "epochs": 3,
  "batch_size": 16,
  "prepare_data_again": 0,
  "need_to_retrain": 1,
  "embedding_model_name": "bert-base-uncased",
  "resulting_model_name": "baseline_model_using_bert",
  "debug": 0,
  "use_cuda": 1
}
'''

## Process

### Data Process
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
Instead of pad seq_len to 128, we firstly sort utterances by length, then pad it to the same length in each batch.

## Inference
The result is identical to that of baseline, yet it is more efficient in training and processing.