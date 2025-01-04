
# Age Prediction Using X-Vectors and MLP

This repository contains the pipeline and scripts to train a Multi-Layer Perceptron (MLP) model for predicting the age of speakers using x-vector features extracted from the Fisher dataset.

## Overview

The project processes audio data from the Fisher dataset to extract x-vector features using SpeechBrain's pre-trained x-vector model. These features are then used to train an MLP model for age prediction.

### Fisher Dataset Statistics

#### Train Dataset
- **Males**: 2,482
- **Females**: 3,956
- **Male Segments**: 28,450
- **Female Segments**: 42,140

#### Development Dataset
- **Males**: 1,000
- **Females**: 1,000
- **Male Segments**: 11,594
- **Female Segments**: 10,775

#### Test Dataset
- **Males**: 1,000
- **Females**: 1,000
- **Male Segments**: 11,743
- **Female Segments**: 10,442

## Feature Extraction

The audio files were preprocessed as follows:
1. Converted to mono channel by separating the speakers.
2. Upsampled from 8kHz to 16kHz.
3. Extracted x-vector features using [SpeechBrain's pre-trained x-vector model](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb).

The `x-vector_extraction.py` script orchestrates the extraction and saves the features in `.npy` format.

## MLP Model Architecture

The MLP model is defined in `MLP_train.py` with the following architecture:
- **Input Size**: 512 (x-vector feature size)
- **Hidden Layers**:
  - Layer 1: 1024 neurons, BatchNorm, SiLU activation, Dropout (0.1)
  - Layer 2: 1024 neurons, BatchNorm, SiLU activation, Dropout (0.2)
  - Layer 3: 512 neurons, BatchNorm, SiLU activation, Dropout (0.1)
- **Output Layer**: 1 neuron for scalar regression (age prediction)

The model was trained using:
- **Optimizer**: Adam
- **Loss Function**: L1 Loss
- **Scheduler**: ReduceLROnPlateau

## Distributed Training

The training process leverages NCCL Distributed Data Parallel (DDP) for efficient multi-GPU training. Key details:
- **Backend**: NCCL
- **Mixed Precision**: Enabled using `torch.cuda.amp.GradScaler`
- **Script**: `mlp_train.sh` for launching training on a GPU cluster.

### Training Environment
- **GPUs**: 2 NVIDIA V100 GPUs
- **CPUs**: 10 cores
- **Memory**: 40 GB
- **Training Duration**: 20 minutes

## Results

### Inference Metrics
- **Train MAE**: 4.2
- **Dev MAE**: 6.67
- **Test MAE**: 6.46


## Usage

### Feature Extraction
Run `x-vector_extraction.py` to preprocess audio files and extract x-vectors:
```bash
python x-vector_extraction.py
```

### Training
Use `mlp_train.sh` to initiate distributed training:
```bash
bash mlp_train.sh
```

### Configuration
Training parameters, such as batch size and learning rate, can be configured directly in `MLP_train.py`.

## Dependencies

Ensure the following packages are installed:
- Python 3.8+
- PyTorch 1.13+
- torchaudio
- numpy
- pandas
- wandb
- tqdm

## Acknowledgments

- [SpeechBrain](https://speechbrain.github.io/) for the pre-trained x-vector model.
- The Fisher corpus for providing the dataset used in this study.
- [Weights & Biases](https://wandb.ai/) for experiment tracking.
