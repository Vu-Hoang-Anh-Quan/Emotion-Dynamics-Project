## Process
### Dataset: DailyDialog
Produce conversation like datasets
```python
{
    "conversation": [
        {"text": str, "speaker": int},
        {"text": str, "speaker": int},
        ...
    ]
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

### Single-utterance classification
- BERT encoder: Encode each utterance with mean pooling then project to 768
- Linear layer: 768 -> 7: linear layer map

### Conversation classification
- BERT encoder: Encode each utterance with mean pooling then project to 768
- Transformer:
    - 8 heads self attention, each having 4 types of masks -> creating 4 outputs -> concatenated -> Linear into 768
    - Feed Forward Network with middle size of 3072 then back to 768
    - Stack again for 8 layers
- Classifier head:
→ Linear (7)
→ logits

### Training
Fine-tune the single-utterance classification, to produce the embedding including BERT and mean pooling. Then, I concatenate it into the main pipeline.

## Hypothesis
Different masking might enables transformer to learns different interaction patterns that can enrich understanding.

## Run result
Single-utterance recognition baseline: Val 0.3506 F1-m, Test 0.4397 F1-m

Setup: 5 epochs for embedding fine-tuning, 30 epochs for whole architecture fine-tuning, dropout rate 0.3, lr 5e-5, batch size 4, unfreeze after 8 epochs, 
achieving 0.3517, 0.4400 F1-m on the validation and test set, respectively.

Setup: 5 epochs for embedding fine-tuning, 30 epochs for whole architecture fine-tuning, dropout rate 0.3, lr 1e-4, batch size 4, unfreeze after 8 epochs, achieving 0.3595, 0.4331 F1-m on the validation and test set, respectively.

## Inference
More complex architecture does not necessarily improve model performance on the F1-score scale, as performances of the architecture and its baseline are near identical. 