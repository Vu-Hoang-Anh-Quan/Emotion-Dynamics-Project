# Baseline with dynamic padding experiment note

# Baseline experiment note

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