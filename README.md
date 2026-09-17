# Modeling Emotional State Dynamics in Conversations

A research project on modeling emotional states in multi-turn conversations, with a focus on how contextual interactions between utterances can improve emotion recognition.

## Overview

This project investigates **Emotion Recognition in Conversations (ERC)** using the DailyDialog dataset.

Rather than treating each utterance as an independent classification problem, the project explores whether explicitly modeling interactions between utterances can produce better representations for emotion recognition.

The main experimental direction is an utterance-level representation and attention architecture in which contextual information is aggregated through self-attention over the utterances in a conversation.

The project is primarily intended as a research and experimentation project. The emphasis is on understanding model behavior through controlled experiments, ablations, and analysis rather than solely optimizing benchmark performance.

## Research Question

The central question is:

> **How should contextual relationships between utterances be modeled to improve emotion recognition in conversations?**

The project explores this question through several stages:

1. Establishing a single-utterance BERT baseline.
2. Investigating alternative utterance representations, including mean pooling.
3. Introducing explicit context modeling between utterances.
4. Investigating recurrent and attention-based approaches to contextual aggregation.
5. Exploring different attention masks to control which utterances can interact.
6. Examining how architectural and training choices affect the resulting representations and performance.

## Dataset and Preprocessing

The project uses **DailyDialog**, a multi-turn dialogue dataset containing conversational utterances annotated with emotion labels.

Each utterance is associated with:

* the utterance text,
* its position within the conversation,
* its speaker,
* an emotion label.

For utterance-level training, utterances are tokenized independently using `bert-base-uncased`, with sequences truncated to a maximum length of 512 tokens and dynamically padded within each batch.

For conversation-level training, the utterances belonging to each conversation are flattened and tokenized together before being reconstructed into a batch with the shape:

```text
[B, T, L]
```

where:

* `B` is the batch size,
* `T` is the maximum number of utterances in a conversation within the batch,
* `L` is the token sequence length.

Conversations with fewer than `T` utterances are padded at the utterance level. A separate utterance mask distinguishes real utterances from padding, while an attention mask identifies valid tokens within each utterance.

Speaker IDs and utterance IDs are retained throughout batching to support conversation-level contextual modeling.

The emotion classification task contains seven emotion classes, including the `Neutral` class.

Model performance is primarily evaluated using **macro-F1 excluding the Neutral class**, following the evaluation convention used in this project.

## Final Experimental Architecture

The final experiment investigates an utterance-level self-attention architecture with multiple attention masks.

The overall pipeline is:

```text
Conversation
    │
    ├── Utterance 1 ── BERT ──┐
    ├── Utterance 2 ── BERT ──┤
    ├── Utterance 3 ── BERT ──┤
    └── ... ────────── BERT ──┘
                              │
                    Utterance representations
                              │
                              ▼
                    Multi-mask self-attention
                              │
                              ▼
                    Contextual representations
                              │
                              ▼
                    Conversation classifier
                              │
                              ▼
                         Emotion label
```

The attention module constructs contextual representations of utterances while controlling the available contextual information through different attention masks.

The final experiment uses multiple masks within the same attention module to represent different contextual scopes.

The masks investigated in the final architecture include:

* **Global mask:** allows attention across all available utterances.
* **Last-$k$ mask:** restricts attention to a local contextual window.
* **Speaker mask:** restricts interactions according to speaker identity.
* **Listener mask:** models interactions from the perspective of the other conversational participant.

The exact mathematical definitions of these masks are documented in the research notes.

## Representation Learning

The project separates utterance-level representation learning from conversation-level context modeling.

First, `bert-base-uncased` is fine-tuned on the utterance-level emotion classification task. The resulting BERT representations are then used as the initial utterance representations for conversation-level modeling.

During the final conversation-level experiment, the BERT encoder is initially frozen for the first **8 epochs**. It is subsequently made trainable, allowing the utterance representations and the contextual attention module to adapt jointly during the later training stages.

This staged training procedure separates the initial optimization of the utterance representations from the subsequent adaptation of those representations to conversational context.

## Experimental Design

Experiments are organized around explicit hypotheses rather than isolated architecture changes.

The research process records:

* the hypothesis being tested,
* the configuration,
* the architectural change,
* the training procedure,
* validation and test results,
* observations,
* interpretation,
* the next experimental step.

The configuration for the Final Experiment is stored separately from the default configuration.

For reproducibility, the configuration used for the Final Experiment is recorded in its experiment log, together with the relevant training and evaluation information.

The earlier experiments were exploratory and were not all maintained as fully reproducible runs. Their notes, logs, and configurations are retained where available, but reconstructing every historical experiment would require recovering numerous intermediate code states and configurations across a large number of commits. Therefore, reproducibility is provided explicitly for the Final Experiment rather than retroactively reconstructed for the entire experimental history.

## Reproducibility

The **Final Experiment** is the reproducible reference experiment archived in this repository.

Its record includes:

* experiment configuration,
* random seed,
* training settings,
* dataset and preprocessing configuration,
* evaluation results,
* the Git commit corresponding to the final implementation.

The Final Experiment corresponds to the latest commit on the `main` branch.

The experiment record follows this structure:

```text
experiments/
└── <experiment-name>/
    ├── log.txt
    └── <experiment-name> - Notes.md
```

The experiment log contains the configuration and information recorded for the final run.

The previous experiments are preserved primarily as a record of the research process. They should not be assumed to be independently reproducible from the current repository state.

## Project Structure

```text
Emotion-Dynamics-Project/
│
├── src/
│   └── Core implementation
│
├── configs/
│   ├── default.json
│   └── final_experiment.json
│
├── experiments/
│   └── Experiment logs and notes
│
├── main.py
├── requirements.txt
├── requirements_cuda.txt
└── README.md
```

## Setup

### CPU

```bash
conda create -n emotion-dynamics python=3.10
conda activate emotion-dynamics

pip install -r requirements.txt
```

### NVIDIA GPU

```bash
conda create -n emotion-dynamics python=3.10
conda activate emotion-dynamics

pip install -r requirements.txt
pip install -r requirements_cuda.txt
```

### Running the Final Experiment

The Final Experiment can be run with:

```bash
python main.py
```

The final experiment configuration is defined in `configs/final_experiment.json`.

## Results

### Baseline

| Model               | Validation Macro-F1 | Test Macro-F1 |
| ------------------- | ------------------: | ------------: |
| BERT + Mean Pooling |              0.3506 |        0.4397 |

### Final Experiment

| Model                                      | Validation Macro-F1 | Test Macro-F1 |
| ------------------------------------------ | ------------------: | ------------: |
| BERT + Mean Pooling + Multi-mask Attention |              0.3595 |        0.4331 |

These results are reported as part of the experimental investigation rather than as a claim of state-of-the-art performance.

## Research Notes

The detailed research process is maintained separately from this README.

The notes document:

* hypotheses,
* failed experiments,
* implementation bugs,
* debugging observations,
* architectural changes,
* ablation studies,
* interpretations of results,
* decisions about subsequent experiments.

The README describes the project and its final methodology, while the research notes preserve the experimental reasoning that led to it.

## Current Status

The current research direction focuses on **utterance-level self-attention with multiple contextual masks** and on understanding whether different interaction structures provide complementary information for emotion recognition.

The project is considered complete in its current form. The Final Experiment is provided as the reproducible reference experiment, while the preceding experiments are retained as a record of the research process.

## Future Work

Potential extensions include:

* further analysis of learned attention patterns,
* controlled ablations of individual attention masks,
* investigation of alternative contextual representations,
* comparison with other conversation-level architectures,
* analysis of errors associated with different emotional transitions.

The author currently has no plans to continue this project, so these are possible research directions rather than planned extensions.

## Research Process

The project follows an iterative experimental process:

```text
Hypothesis
    ↓
Experiment
    ↓
Result inspection
    ↓
Debugging / analysis
    ↓
Interpretation
    ↓
Next hypothesis
```

The objective is not simply to accumulate model variants, but to understand which assumptions about conversational context lead to useful representations and why.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for the full license text.

Copyright (c) 2026 Vu Hoang Anh Quan