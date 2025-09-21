# Small Language Model (SLM) Training

This project implements a Small Language Model using the TinyStories dataset and GPT architecture.

## Features

- **Dataset**: TinyStories dataset for training
- **Architecture**: GPT-style transformer with 6 layers, 6 heads, 384 embedding dimensions
- **Training**: Mixed precision training with gradient accumulation
- **Optimization**: AdamW optimizer with cosine annealing learning rate schedule
- **Inference**: Text generation with temperature and top-k sampling

## Model Specifications

- **Vocabulary Size**: 50,257 (GPT-2 tokenizer)
- **Context Window**: 128 tokens
- **Layers**: 6 transformer blocks
- **Attention Heads**: 6
- **Embedding Dimension**: 384
- **Dropout**: 0.1

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete training pipeline:

```bash
python slm_training.py
```

This will:
1. Download and tokenize the TinyStories dataset
2. Train the model for 20,000 iterations
3. Save the best model checkpoint
4. Generate sample text outputs
5. Plot training/validation losses

## Training Configuration

- **Learning Rate**: 1e-4 with warmup and cosine annealing
- **Batch Size**: 32
- **Gradient Accumulation**: 32 steps
- **Mixed Precision**: Automatic (bfloat16/float16)
- **Gradient Clipping**: 0.5 max norm

## Output Files

- `best_model_params.pt`: Best model checkpoint
- `train.bin`: Tokenized training data
- `validation.bin`: Tokenized validation data
- `training_losses.png`: Training/validation loss plot

## Model Performance

The model will generate coherent short stories similar to the TinyStories dataset style. Sample outputs include continuation of prompts like "Once upon a time..." with child-friendly content.

## Hardware Requirements

- **GPU**: Recommended (CUDA support)
- **RAM**: 8GB+ recommended
- **Storage**: ~2GB for dataset and checkpoints
