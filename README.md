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

## Goals

- [ ] Build baseline models for emotion classification
- [ ] Model temporal dependencies (RNN / Transformer)
- [ ] Experiment with contextual embeddings
- [ ] Evaluate on conversational datasets (e.g., MELD)

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
Separate each uterrance with labeled emotoin from dataset DailyDialogue
Fine-tuning a BERT embedding model with a classifier head to classify each utterance with their emotion.

## Future Work

- Adding context into data representation
- Adding context processing in pipeline
- Adding a transformer to better process sequential context of utterances

## Current development
Input: change from one utterance to sequence of utterances
Output: emotion per utterance
Constraint to be awared of: limited context window
Goal: find a suitable way to include context in data and processing while under constraint but still provides great information for classification