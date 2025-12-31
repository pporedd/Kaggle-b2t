# Summary of kaggle-b2t Repository

## Overview
This repository contains the code and resources for the paper **"An Accurate and Rapidly Calibrating Speech Neuroprosthesis"** (Card et al., 2024, *NEJM*). It also serves as the baseline code for the **Brain-to-Text '25 Competition** on Kaggle.

The project implements a speech neuroprosthesis that decodes neural activity into text using a Recurrent Neural Network (RNN) and an N-gram language model with LLM rescoring.

## Key Directories

### 1. `model_training/`
Contains the core deep learning code (PyTorch).
-   **Model**: A custom RNN with 5 GRU layers (768 units each) and day-specific input layers.
-   **Key Scripts**:
    -   `train_model.py`: Trains the baseline RNN on the provided dataset.
    -   `evaluate_model.py`: Evaluates the model on validation/test sets. It infers phonemes and sends them to the language model via Redis.
    -   `rnn_model.py`: Defines the RNN architecture.
    -   `rnn_args.yaml`: Configuration file for hyperparameters.

### 2. `language_model/`
A standalone component for decoding phonemes into text.
-   **Functionality**: Implements an N-gram language model (1-gram, 3-gram, or 5-gram) and optional rescoring using **Facebook OPT-6.7b**.
-   **Communication**: Runs as a separate process and communicates with the main evaluation script via **Redis**.
-   **Requirements**: Requires a separate conda environment (`b2txt25_lm`) and potentially large RAM/VRAM depending on the n-gram order and LLM usage.

### 3. `analyses/`
Contains Jupyter notebooks for reproducing the paper's figures.
-   `figure_2.ipynb`: Online Copy Task results.
-   `figure_4.ipynb`: Conversation Mode data.

### 4. `data/`
Directory for datasets (hosted on Dryad). Expected structure includes:
-   `t15_copyTask_neuralData`: Neuaral data (HDF5).
-   `t15_pretrained_rnn_baseline`: Pretrained model checkpoints.
-   `t15_copyTask.pkl`, `t15_personalUse.pkl`: Analysis data files.

## Utilities & Others
-   `utils/`: General utility functions.
-   `runtime/`, `srilm-1.7.3/`, `wenet/`: Dependencies and build artifacts for the language model (Kaldi-based).
-   `dataset.py`: Handles data loading and augmentation (noise, temporal jitter).

## Setup & Requirements
The repository requires a specific setup to run the full pipeline:
1.  **Redis**: Must be installed and running (`redis-server`) to bridge the RNN and Language Model.
2.  **Environments**:
    -   `b2txt25`: Main environment for model training/eval (setup via `./setup.sh`).
    -   `b2txt25_lm`: Separate environment for the language model (setup via `./setup_lm.sh`).
3.  **Hardware**: Tested on Ubuntu 22.04 with dual RTX 4090s. High RAM is needed for 5-gram LMs.
