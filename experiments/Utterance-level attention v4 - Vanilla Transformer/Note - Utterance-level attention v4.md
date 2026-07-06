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
- BERT encoder: Encode each utterance with mean pooling then project to 256
- Linear layer: 256 -> 7: linear layer map

### Conversation classification
- BERT encoder: Encode each utterance with mean pooling then project to 256
- Transformer:
    - 8 heads self attention
    - Feed Forward Network with middle size of 3072 then back to 768
    - Stack again for 4 layers
- Classifier head:
→ Linear (7)
→ logits

### Training
Fine-tune the single-utterance classification, specifically BERT first, before concatenate it into the main pipeline.

## Hypothesis
Applying both multi-head attention and stacking layers of attention and feed forward can produce better result in classification, compared to just a single-head single-layer self-attention.

## Run result
Single-utterance pipeline: Val 9.3488 F1-m, Test 0.4363 F1-m

Setup: 5 epochs for embedding fine-tuning, 30 epochs for whole architecture fine-tuning, dropout rate 0.2, lr 1e-4, batch size 4
- Best result: Val 0.3658 F1-m, Test 0.4481 F1-m

Setup: 5 epochs for embedding fine-tuning, 30 epochs for whole architecture fine-tuning, dropout rate 0.3, lr 5e-5, batch size 4
- Best result: Val 0.3803 F1-m, Test 0.4375 F1-m

## Inference
There is a persistent bug in building dataloader from GRU experiment until now that cause bottleneck results. After finding it, the score returned to normal