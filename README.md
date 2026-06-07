# Modeling Emotional State Dynamics in Conversations

## Overview

This project studies how emotional states evolve over the course of a conversation.

The goal is to model temporal emotional transitions using machine learning methods, 
with a focus on capturing dependencies between utterances.

This project is currently in early development.

## Motivation

Understanding emotional dynamics is important for applications such as:
- mental health support systems
- conversational AI
- dialogue understanding
- human-behavorial understanding

This project explores how models can track and predict emotional shifts over time.

## Setup
### CPU
```bash
conda create -n emotion-dynamics python=3.10
pip install -r requirements.txt
```
### GPU
```bash
conda create -n emotion-dynamics python=3.10
pip install -r requirements.txt
pip install -r requirements_cuda.txt
```

## Project Structure

Emotion Dynamics Project/
- src/            # core code
- configs/        # experiment configs
- experiments/    # logs and notes
- notebooks/      # exploration
- README.md

## Current Status

Having the baseline model:
- Separate each uterrance with labeled emotion from dataset DailyDialogue
- Fine-tuning a BERT embedding model with a classifier head to classify each utterance with their emotion.

## Research Questions

- How learned pooling can outperform BERT CLS
- How context modelling will improve performance?
- Can temporal models that learn the shifts between emotions outperform static classifications?
- How graph neural network differs from transformers, and how combining them be better than isolated?

## Possible future work
- Add logging
- Remember to not load tokenizer the second time in both training pipelines
- Combine config into 1
- Just in case, maybe LR was too high for attention that it collapse completely. 5e-5 or lower should be aimed for lr, not 1e-3

- Experiment: Remember to freeze BERT for k=3 epochs when fine-tune whole, change the freeze function to set_trainable_layer(bert_model, k) and remember to rebuild the optimizer when do that
### Replace BERT CLS with learned pooling
Hypothesis: As CLS is optimized for next-sentence prediction, it is not adapted to emotion classification. Therefore, replacing it with a learned pooling that look for specific richful tokens will further enrich the representation of each utterance.

### Separated training
Firstly train the BERT for single-utterance emotion recognition first, before combining the pipeline.

### Confirm context processing is working in attention
Inspect attention weight so that all the probabilities do not go just from the utterance itself

### Add speaker-aware information
This can be in the form of speaker embedding and/or speaker masking

### Use focal loss

## Current development
Tokenize -> Embed utterances into vectors -> GRU -> emotion