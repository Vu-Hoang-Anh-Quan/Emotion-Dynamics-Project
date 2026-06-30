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

### Try stacking multiple attention layers -> encourages them to diverge

### Try a another loss that encourages sharp attention probability
Inverse participation ratio
- Build a new loss
- Change config: new loss
- Change loop pipeline: apply the right loss
- More config: lambda rate

### Try GCN with relation types
Proposed architecture: 16-relationship corpus -> pre-trained
Build a GCN with pytorch with these types

### Add speaker-aware information
This can be in the form of speaker embedding and/or speaker masking

### Replace BERT CLS with learned pooling
Hypothesis: As CLS is optimized for next-sentence prediction, it is not adapted to emotion classification. Therefore, replacing it with a learned pooling that look for specific richful tokens will further enrich the representation of each utterance.

### Use focal loss

## Current development
Firstly pretrain the embedding, then freeze it.

Tokenize -> Embed utterances into vectors -> self-attention -> emotion