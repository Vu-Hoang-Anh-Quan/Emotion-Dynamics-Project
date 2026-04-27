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

- Instead of concatenating context in string form, how about also using the hidden state of last utterance as context for current utterance?
- How dropout rate impact training efficiency?
- How weight decay (around 0.01) impact fine-tuning?
- Can making the classifying head deeper improves performance?
- How learning rate increase over time affect BERT?
- What happens if we freeze BERT and only fine-tune the classifying head?
- Does learning loss that accounts for the "Neutral" label from DailyDialog dataset reduce learning efficiency, or weighted loss improve training efficiency?

- How context modelling will improve performance?
- Can temporal models that learn the shifts between emotions outperform static classifications?
- How graph neural network differs from transformers, and how combining them be better than isolated?

## Next work
### Deeper head
Making the head deeper with another layer and layer norm: CLS (768) -> hidden layer (256) -> LayerNorm -> logits (7)

Hypothesis: 
This will certainly makes the model able to recognize more complex samples, especially with the adding of context that it needs to process much more information. However, it can also more prone to overfitting.

### Freeze BERT
Do not fine-tune BERT also, instead, focusing on the now deep and capable head. Also include another layer of 128 after 256.

Hypothesis: 
This will increase training speed greatly, and also allows the model to focus more on the classification after having semantic from BERT. However, without fine-tuning BERT, we will likely see a drop in performance, as BERT is not pretrained for emotion recognizing.

## Possible Future Work

- Adding a transformer to better process sequential context of utterances
- Adding a graph neural network to better model the relationships between utterances
- Changing the training process to let model to further detect other labels than the "Neutral" that dominates the dataset

## Current development
Input: change from one utterance to sequence of utterances

Output: emotion per utterance

Constraint to be awared of: limited context window

Goal: find a suitable way to include context in data and processing while under constraint but still provides great information for classification