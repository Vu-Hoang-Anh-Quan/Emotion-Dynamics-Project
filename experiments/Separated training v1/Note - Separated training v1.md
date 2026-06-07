## Process
### Dataset: DailyDialog
Separate each conversation into training samples of (context → current utterance):

```python
{
    "context": [
        {"text": str, "speaker": int},  # (i-1), most recent
        {"text": str, "speaker": int},  # (i-2)
        ...
    ],
    "current": {
        "text": str,        # target utterance
        "speaker": int,     # 0 / 1
        "emotion": int      # label (0–6)
    }
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
- BERT encoder: Encode each utterance with mean pooling
- 768 -> 7: linear layer map

### Conversation classification
- BERT encoder: Encode each utterance with mean pooling
- Self-attention layer + residual: 128
    - Casual mask to ensure temporal order
- Classifier head:
→ Linear (128)
→ LayerNorm
→ ReLU
→ Dropout
→ Linear (7)
→ logits

### Training
Fine-tune the single-utterance classification, specifically BERT first, before concatenate it into the main pipeline

## Hypothesis
Hypothesis: Through separated training, we hope that BERT will better capture information from each utterance, therefore provide much more semantics when combine with context during full training, improving the final score, at least by a margin compared to baseline v4. 
However, the problem might still persists as BERT should be freezed or reduce learning rate so that it won't be impacted by the noisy gradient when training the whole pipeline.

## Run result
```
Epoch 1/5
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5449/5449 [02:35<00:00, 35.06it/s]
Class 0: 77.5313
Class 1: 0.9047
Class 2: 0.0620
Class 3: 0.2850
Class 4: 17.7717
Class 5: 1.2021
Class 6: 2.2432
Train Loss: 1.0655
Val Loss:   0.7111 | Val Acc: 0.8446 | Val F1-score macro: 0.4155 | Val F1-score macro non-Neutral: 0.3328
Current model saved to directory /home/Quan/Desktop/Emotion-Dynamics-Project/saved_models/utterance_classifier.pt

Epoch 2/5
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5449/5449 [02:35<00:00, 35.10it/s]
Class 0: 73.8629
Class 1: 0.6197
Class 2: 0.2231
Class 3: 0.0372
Class 4: 18.8127
Class 5: 4.2137
Class 6: 2.2308
Train Loss: 0.8656
Val Loss:   0.8306 | Val Acc: 0.8173 | Val F1-score macro: 0.4084 | Val F1-score macro non-Neutral: 0.3278

Epoch 3/5
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5449/5449 [02:35<00:00, 35.13it/s]
Class 0: 75.5732
Class 1: 1.7102
Class 2: 0.2231
Class 3: 0.0124
Class 4: 17.6602
Class 5: 2.9991
Class 6: 1.8218
Train Loss: 0.7408
Val Loss:   0.8375 | Val Acc: 0.8315 | Val F1-score macro: 0.3818 | Val F1-score macro non-Neutral: 0.2950

Epoch 4/5
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5449/5449 [02:35<00:00, 35.13it/s]
Class 0: 79.2911
Class 1: 0.6816
Class 2: 0.1239
Class 3: 0.0744
Class 4: 17.0157
Class 5: 1.1897
Class 6: 1.6235
Train Loss: 0.6087
Val Loss:   1.0339 | Val Acc: 0.8543 | Val F1-score macro: 0.4114 | Val F1-score macro non-Neutral: 0.3272

Epoch 5/5
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5449/5449 [02:34<00:00, 35.18it/s]
Class 0: 80.9270
Class 1: 0.6568
Class 2: 0.1115
Class 3: 0.0744
Class 4: 15.3303
Class 5: 1.1154
Class 6: 1.7846
Train Loss: 0.4920
Val Loss:   1.3180 | Val Acc: 0.8664 | Val F1-score macro: 0.4183 | Val F1-score macro non-Neutral: 0.3339
Current model saved to directory /home/Quan/Desktop/Emotion-Dynamics-Project/saved_models/utterance_classifier.pt
Class 0: 75.5943
Class 1: 1.1628
Class 2: 0.2842
Class 3: 0.1680
Class 4: 18.6176
Class 5: 1.7700
Class 6: 2.4031
Utterance Pipeline - Test Loss: 0.6862 | Test Acc: 0.8149 | Test F1-score macro: 0.4806 | Test F1-score macro non-Neutral: 0.4127

Running conversation pipeline...
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 19676.25it/s]
BertModel LOAD REPORT from: bert-base-uncased
Key                                        | Status     |  | 
-------------------------------------------+------------+--+-
cls.predictions.bias                       | UNEXPECTED |  | 
cls.seq_relationship.bias                  | UNEXPECTED |  | 
cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
cls.seq_relationship.weight                | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Using device: cuda | AMP: True
tensor([0.0074, 0.6452, 1.7610, 3.6546, 0.0477, 0.5506, 0.3335],
       device='cuda:0')

Epoch 1/10
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2780/2780 [02:17<00:00, 20.15it/s]
Class 0: 84.9548
Class 1: 0.0496
Class 2: 0.0000
Class 3: 0.0000
Class 4: 13.1491
Class 5: 1.3880
Class 6: 0.4585
Train Loss: 1.7142
Val Loss:   1.5300 | Val Acc: 0.7961 | Val F1-score macro: 0.1859 | Val F1-score macro non-Neutral: 0.0693
Current model saved to directory /home/Quan/Desktop/Emotion-Dynamics-Project/saved_models/conversation_classifier.pt

Epoch 2/10
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2780/2780 [02:18<00:00, 20.06it/s]
Class 0: 70.7647
Class 1: 0.5081
Class 2: 0.0000
Class 3: 0.0000
Class 4: 24.6995
Class 5: 1.2517
Class 6: 2.7761
Train Loss: 1.6713
Val Loss:   1.5678 | Val Acc: 0.6835 | Val F1-score macro: 0.1868 | Val F1-score macro non-Neutral: 0.0832
Current model saved to directory /home/Quan/Desktop/Emotion-Dynamics-Project/saved_models/conversation_classifier.pt

Epoch 3/10
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2780/2780 [02:18<00:00, 20.06it/s]
Class 0: 83.2445
Class 1: 0.6568
Class 2: 0.0000
Class 3: 0.0000
Class 4: 15.1816
Class 5: 0.2355
Class 6: 0.6816
Train Loss: 1.6676
Val Loss:   1.5658 | Val Acc: 0.7841 | Val F1-score macro: 0.2010 | Val F1-score macro non-Neutral: 0.0882
Current model saved to directory /home/Quan/Desktop/Emotion-Dynamics-Project/saved_models/conversation_classifier.pt

Epoch 4/10
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2780/2780 [02:18<00:00, 20.09it/s]
Class 0: 81.4227
Class 1: 0.5825
Class 2: 0.0000
Class 3: 0.0000
Class 4: 13.5457
Class 5: 3.0859
Class 6: 1.3632
Train Loss: 1.6591
Val Loss:   1.5369 | Val Acc: 0.7675 | Val F1-score macro: 0.2030 | Val F1-score macro non-Neutral: 0.0922
Current model saved to directory /home/Quan/Desktop/Emotion-Dynamics-Project/saved_models/conversation_classifier.pt

Epoch 5/10
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2780/2780 [02:18<00:00, 20.05it/s]
Class 0: 88.5116
Class 1: 0.3594
Class 2: 0.0000
Class 3: 0.0000
Class 4: 9.4064
Class 5: 0.9419
Class 6: 0.7808
Train Loss: 1.6581
Val Loss:   1.5802 | Val Acc: 0.8238 | Val F1-score macro: 0.2027 | Val F1-score macro non-Neutral: 0.0860

Epoch 6/10
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2780/2780 [02:18<00:00, 20.13it/s]
Class 0: 84.7565
Class 1: 1.2517
Class 2: 0.0000
Class 3: 0.0000
Class 4: 12.1948
Class 5: 0.8923
Class 6: 0.9047
Train Loss: 1.6375
Val Loss:   1.5387 | Val Acc: 0.7990 | Val F1-score macro: 0.2092 | Val F1-score macro non-Neutral: 0.0961
Current model saved to directory /home/Quan/Desktop/Emotion-Dynamics-Project/saved_models/conversation_classifier.pt

Epoch 7/10
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2780/2780 [02:17<00:00, 20.22it/s]
Class 0: 82.3522
Class 1: 0.6940
Class 2: 0.0000
Class 3: 0.0000
Class 4: 16.1978
Class 5: 0.0000
Class 6: 0.7560
Train Loss: 1.6417
Val Loss:   1.5959 | Val Acc: 0.7787 | Val F1-score macro: 0.1985 | Val F1-score macro non-Neutral: 0.0859

Epoch 8/10
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2780/2780 [02:18<00:00, 20.11it/s]
Class 0: 77.9527
Class 1: 0.6320
Class 2: 0.0000
Class 3: 0.0000
Class 4: 19.7174
Class 5: 0.7560
Class 6: 0.9419
Train Loss: 1.6439
Val Loss:   1.5688 | Val Acc: 0.7445 | Val F1-score macro: 0.1991 | Val F1-score macro non-Neutral: 0.0904

Epoch 9/10
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2780/2780 [02:17<00:00, 20.21it/s]
Class 0: 83.1206
Class 1: 0.8923
Class 2: 0.0248
Class 3: 0.0000
Class 4: 14.1529
Class 5: 0.4833
Class 6: 1.3261
Train Loss: 1.6351
Val Loss:   1.5596 | Val Acc: 0.7858 | Val F1-score macro: 0.2053 | Val F1-score macro non-Neutral: 0.0930

Epoch 10/10
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2780/2780 [02:18<00:00, 20.09it/s]
Class 0: 77.3206
Class 1: 1.4128
Class 2: 0.2603
Class 3: 0.0000
Class 4: 18.3170
Class 5: 1.3385
Class 6: 1.3508
Train Loss: 1.6311
Val Loss:   1.5309 | Val Acc: 0.7401 | Val F1-score macro: 0.2036 | Val F1-score macro non-Neutral: 0.0961
Class 0: 80.9819
Class 1: 1.5762
Class 2: 0.0517
Class 3: 0.0000
Class 4: 13.6176
Class 5: 2.2739
Class 6: 1.4987
Conversation Pipeline - Test Loss: 1.0656 | Test Acc: 0.7214 | Test F1-score macro: 0.2066 | Test F1-score macro non-Neutral: 0.1013
Run completed successfully.
```

## Inference
It seems that due to the nature of emotion being the combination of all emotional semantics among the utterance, mean pooling outperformed CLS on the same benchmask, reaching 0.1 instead of just 0.08~0.09. Moreover, it can be seen that the model was able to learn the second least common emotion - class 2. This shows that mean pooling and other custom pooling has the potential to outperform CLS.

However, the bottleneck is yet to be resolved, as score is still ridiculously small compared to baseline v2. The problem now is very likely to lie in the gradients flowing from loss, through attention to BERT, is too far away, making it weak and noisy. 
Therefore, the next work will lies on separated training session, with single-utterance recognition part being trained first, before concatenating into the main pipeline.