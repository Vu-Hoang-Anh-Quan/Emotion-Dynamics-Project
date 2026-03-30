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
### Model

### Training

## Result
