# SLM Experiment Framework

This framework allows you to systematically experiment with different configurations of your Small Language Model, including vocabulary sizes, embedding dimensions, and training data variations.

## Overview

The experiment framework consists of several components:

- **`experiment_config.py`** - Defines all experiment configurations
- **`run_experiment.py`** - Runs individual experiments
- **`run_batch_experiments.py`** - Runs multiple experiments and compares results
- **`experiment_runner.py`** - Interactive experiment runner

## Available Experiments

### Baseline Configuration
- **`baseline`** - Current best configuration (vocab: 50,257, emb: 256, layers: 4)

### Vocabulary Size Experiments
- **`vocab_small`** - Reduced vocabulary (10,000 tokens)
- **`vocab_medium`** - Medium vocabulary (25,000 tokens)

### Embedding Dimension Experiments
- **`embd_small`** - Small embeddings (128 dimensions)
- **`embd_large`** - Large embeddings (512 dimensions)
- **`embd_xlarge`** - Extra large embeddings (768 dimensions)

### Architecture Experiments
- **`arch_deep`** - Deeper model (8 layers)
- **`arch_wide`** - Wider attention (8 heads)

### Training Data Experiments
- **`data_small`** - Use only 10% of training data
- **`data_medium`** - Use 50% of training data

### Learning Rate Experiments
- **`lr_high`** - Higher learning rate (5e-4)
- **`lr_low`** - Lower learning rate (5e-5)

### Optimized Configurations
- **`optimal_small`** - Optimized small model
- **`optimal_fast`** - Fast training optimized model

## Quick Start

### 1. Run Individual Experiments

```bash
# Interactive mode
python experiment_runner.py

# Run specific experiment
python experiment_runner.py run baseline

# Show experiment details
python experiment_runner.py details baseline

# List all experiments
python experiment_runner.py list
```

### 2. Run Batch Experiments

```bash
# Run quick experiments (for testing)
python run_batch_experiments.py quick

# Run vocabulary experiments
python run_batch_experiments.py vocab

# Run embedding experiments
python run_batch_experiments.py embedding

# Run all experiments
python run_batch_experiments.py all

# Compare existing results
python run_batch_experiments.py compare
```

### 3. Run Single Experiment Directly

```bash
python run_experiment.py baseline
```

## Experiment Results

Each experiment generates:

- **Model checkpoint**: `best_model_{experiment_name}.pt`
- **Results JSON**: `experiments/{experiment_name}_results.json`
- **Training plot**: `experiments/{experiment_name}_training_losses.png`
- **Tokenized data**: `train_{experiment_name}.bin`, `validation_{experiment_name}.bin`

## Batch Results

Batch experiments create:

- **Batch results**: `experiments/batch_results.json`
- **Comparison table**: `experiments/experiment_comparison.csv`
- **Comparison plots**: `experiments/experiment_comparison.png`

## Understanding Results

### Key Metrics

- **Total Parameters**: Model size in parameters
- **Training Time**: Time to complete training
- **Final Train Loss**: Training loss at end of training
- **Final Val Loss**: Validation loss at end of training
- **Best Val Loss**: Best validation loss achieved during training

### Interpretation

- **Lower validation loss** = Better model performance
- **Fewer parameters** = Smaller, faster model
- **Shorter training time** = More efficient training
- **Vocabulary size** affects model capacity and training speed
- **Embedding dimension** affects model expressiveness
- **Dataset size** affects generalization ability

## Custom Experiments

To create your own experiment:

1. Edit `experiment_config.py`
2. Add your configuration to `EXPERIMENT_CONFIGS`
3. Run with: `python run_experiment.py your_experiment_name`

Example:
```python
"my_experiment": ExperimentConfig(
    name="my_experiment",
    vocab_size=30000,
    n_embd=512,
    n_layer=6,
    n_head=8,
    block_size=128,
    batch_size=32,
    learning_rate=2e-4,
    max_iters=3000,
    description="My custom experiment"
)
```

## Hardware Requirements

- **CPU**: All experiments run on CPU for Windows compatibility
- **RAM**: 4-8GB recommended
- **Storage**: ~1-2GB per experiment
- **Time**: 10-60 minutes per experiment depending on configuration

## Tips for Experimentation

### Vocabulary Size
- Smaller vocabularies train faster but may have limited expressiveness
- Larger vocabularies need more data and training time
- Consider your specific use case when choosing vocabulary size

### Embedding Dimensions
- Larger embeddings can capture more complex patterns
- Smaller embeddings are faster and use less memory
- Rule of thumb: embedding_dim = vocab_size^0.25 * some_factor

### Training Data
- More data generally leads to better generalization
- Smaller datasets train faster but may overfit
- Consider data quality vs quantity trade-offs

### Architecture
- More layers = more capacity but slower training
- More attention heads = better attention patterns
- Balance capacity with training efficiency

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or embedding dimension
2. **Slow Training**: Reduce max_iters or use smaller model
3. **Poor Results**: Try different learning rates or more data
4. **File Not Found**: Ensure dataset is downloaded first

### Performance Tips

- Start with quick experiments to test configurations
- Use smaller datasets for initial testing
- Monitor training curves for overfitting
- Compare multiple runs for statistical significance

## Example Workflow

1. **Start with baseline**: `python experiment_runner.py run baseline`
2. **Test vocabulary sizes**: `python run_batch_experiments.py vocab`
3. **Test embedding dimensions**: `python run_batch_experiments.py embedding`
4. **Compare results**: `python run_batch_experiments.py compare`
5. **Create custom experiment** based on findings
6. **Run final comparison** with best configurations

## File Structure

```
SLM/
├── experiment_config.py          # Experiment definitions
├── run_experiment.py             # Single experiment runner
├── run_batch_experiments.py      # Batch experiment runner
├── experiment_runner.py          # Interactive runner
├── experiments/                  # Results directory
│   ├── batch_results.json       # Batch experiment results
│   ├── experiment_comparison.csv # Comparison table
│   ├── experiment_comparison.png # Comparison plots
│   └── {experiment_name}_*      # Individual experiment files
└── EXPERIMENT_README.md          # This file
```

## Next Steps

After running experiments:

1. **Analyze results** using comparison plots and tables
2. **Identify best configurations** for your use case
3. **Create optimized models** based on findings
4. **Document insights** for future reference
5. **Share results** with the community

Happy experimenting! 🚀

