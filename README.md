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

- How dropout rate impact training efficiency?
- How weight decay (around 0.01) impact fine-tuning?
- How lowering learning rate to 1e-5 increase training efficiency?
- How learning rate increase over time affect BERT?
- Can making the classifying head deeper improves performance?
- What happens if we freeze BERT and only fine-tune the classifying head?
- Does learning loss that accounts for the "Neutral" label from DailyDialog dataset reduce learning efficiency, or weighted loss improve training efficiency?

- How context modelling will improve performance?
- Can temporal models that learn the shifts between emotions outperform static classifications?
- How graph neural network differs from transformers, and how combining them be better than isolated?

## Possible Future Work

- Change the way of calculating F1-score to account for DailyDialog

- Produce speaker-aware tokenized data
- Adding context into data representation
- Adding context processing in pipeline
- Adding a transformer to better process sequential context of utterances
- Adding a graph neural network to better model the relationships between utterances
- Changing the training process to let model to further detect other labels than the "Neutral" that dominates the dataset

## Current development
Input: change from one utterance to sequence of utterances

Output: emotion per utterance

Constraint to be awared of: limited context window

Goal: find a suitable way to include context in data and processing while under constraint but still provides great information for classification