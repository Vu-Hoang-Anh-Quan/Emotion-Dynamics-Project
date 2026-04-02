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
Using BERT embedding model, with a classifier head added.

### Training
Using Cross Entropy loss, 3 epochs, batch size 16 fine-tuning both the embedding and classifier head. 

## Result
Epoch 1/3
Train Loss: 0.6518
Val Loss:   6.4155 | Val Acc: 0.0004
Current model saved to directory /content/drive/MyDrive/Emotional Dynamics Project/saved_models/baseline_model_using_bert.pt

Epoch 2/3
100% 5449/5449 [10:25<00:00,  8.72it/s]
Train Loss: 0.6527
Val Loss:   0.4834 | Val Acc: 0.8809
Current model saved to directory /content/drive/MyDrive/Emotional Dynamics Project/saved_models/baseline_model_using_bert.pt

Epoch 3/3
100% 5449/5449 [10:25<00:00,  8.71it/s]
Train Loss: 0.6269
Val Loss:   0.4993 | Val Acc: 0.8809
Final test loss: 0.6662846877555216
Final test accuracy: 0.8166666666666667

## Inference
It can be seen that the learning curve goes up on the third epoch. It can be seen that 0.67 on loss and 0.82 on accuraccy is the limit of this model

# Future work
Complete logging pipeline
Add context in data representation.
Add context processing in model (attention, sliding-window, etc.)