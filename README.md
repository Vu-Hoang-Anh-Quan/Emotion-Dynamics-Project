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
```bash
conda create -n emotion-dynamics python=3.10
pip install -r requirements.txt
```

## Project Structure

Emotion Dynamics Project/
- src/            # core code
- configs/        # experiment configs
- experiments/    # logs and notes
- notebooks/      # exploration
- README.md

## Current Status

Developing model:
- Embedding with a classifier head
- Train model

## Future Work

- Implement initial baseline model
- Add training pipeline
- Design evaluation metrics