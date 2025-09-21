#!/usr/bin/env python3
"""
Experiment Configuration Framework for SLM
Allows systematic testing of different model configurations
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import json

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    name: str
    vocab_size: int
    n_embd: int
    n_layer: int
    n_head: int
    block_size: int
    batch_size: int
    learning_rate: float
    max_iters: int
    dropout: float = 0.1
    bias: bool = True
    dataset_subset: float = 1.0  # Fraction of dataset to use (1.0 = full dataset)
    description: str = ""

# Define different experiment configurations
EXPERIMENT_CONFIGS = {
    # Baseline configuration (current best)
    "baseline": ExperimentConfig(
        name="baseline",
        vocab_size=50257,
        n_embd=256,
        n_layer=4,
        n_head=4,
        block_size=64,
        batch_size=16,
        learning_rate=1e-4,
        max_iters=2000,
        description="Current baseline configuration"
    ),
    
    # Vocabulary size experiments
    "vocab_small": ExperimentConfig(
        name="vocab_small",
        vocab_size=10000,  # Reduced vocabulary
        n_embd=256,
        n_layer=4,
        n_head=4,
        block_size=64,
        batch_size=16,
        learning_rate=1e-4,
        max_iters=2000,
        description="Small vocabulary (10K tokens)"
    ),
    
    "vocab_medium": ExperimentConfig(
        name="vocab_medium",
        vocab_size=25000,  # Medium vocabulary
        n_embd=256,
        n_layer=4,
        n_head=4,
        block_size=64,
        batch_size=16,
        learning_rate=1e-4,
        max_iters=2000,
        description="Medium vocabulary (25K tokens)"
    ),
    
    # Embedding dimension experiments
    "embd_small": ExperimentConfig(
        name="embd_small",
        vocab_size=50257,
        n_embd=128,  # Smaller embeddings
        n_layer=4,
        n_head=4,
        block_size=64,
        batch_size=16,
        learning_rate=1e-4,
        max_iters=2000,
        description="Small embedding dimension (128)"
    ),
    
    "embd_large": ExperimentConfig(
        name="embd_large",
        vocab_size=50257,
        n_embd=512,  # Larger embeddings
        n_layer=4,
        n_head=4,
        block_size=64,
        batch_size=16,
        learning_rate=1e-4,
        max_iters=2000,
        description="Large embedding dimension (512)"
    ),
    
    "embd_xlarge": ExperimentConfig(
        name="embd_xlarge",
        vocab_size=50257,
        n_embd=768,  # Much larger embeddings
        n_layer=4,
        n_head=4,
        block_size=64,
        batch_size=16,
        learning_rate=1e-4,
        max_iters=2000,
        description="Extra large embedding dimension (768)"
    ),
    
    # Architecture experiments
    "arch_deep": ExperimentConfig(
        name="arch_deep",
        vocab_size=50257,
        n_embd=256,
        n_layer=8,  # More layers
        n_head=4,
        block_size=64,
        batch_size=16,
        learning_rate=1e-4,
        max_iters=2000,
        description="Deeper architecture (8 layers)"
    ),
    
    "arch_wide": ExperimentConfig(
        name="arch_wide",
        vocab_size=50257,
        n_embd=256,
        n_layer=4,
        n_head=8,  # More attention heads
        block_size=64,
        batch_size=16,
        learning_rate=1e-4,
        max_iters=2000,
        description="Wider attention (8 heads)"
    ),
    
    # Training data experiments
    "data_small": ExperimentConfig(
        name="data_small",
        vocab_size=50257,
        n_embd=256,
        n_layer=4,
        n_head=4,
        block_size=64,
        batch_size=16,
        learning_rate=1e-4,
        max_iters=2000,
        dataset_subset=0.1,  # Use only 10% of data
        description="Small dataset (10% of training data)"
    ),
    
    "data_medium": ExperimentConfig(
        name="data_medium",
        vocab_size=50257,
        n_embd=256,
        n_layer=4,
        n_head=4,
        block_size=64,
        batch_size=16,
        learning_rate=1e-4,
        max_iters=2000,
        dataset_subset=0.5,  # Use 50% of data
        description="Medium dataset (50% of training data)"
    ),
    
    # Learning rate experiments
    "lr_high": ExperimentConfig(
        name="lr_high",
        vocab_size=50257,
        n_embd=256,
        n_layer=4,
        n_head=4,
        block_size=64,
        batch_size=16,
        learning_rate=5e-4,  # Higher learning rate
        max_iters=2000,
        description="High learning rate (5e-4)"
    ),
    
    "lr_low": ExperimentConfig(
        name="lr_low",
        vocab_size=50257,
        n_embd=256,
        n_layer=4,
        n_head=4,
        block_size=64,
        batch_size=16,
        learning_rate=5e-5,  # Lower learning rate
        max_iters=2000,
        description="Low learning rate (5e-5)"
    ),
    
    # Combined experiments
    "optimal_small": ExperimentConfig(
        name="optimal_small",
        vocab_size=25000,
        n_embd=384,
        n_layer=6,
        n_head=6,
        block_size=64,
        batch_size=16,
        learning_rate=1e-4,
        max_iters=2000,
        description="Optimized small model"
    ),
    
    "optimal_fast": ExperimentConfig(
        name="optimal_fast",
        vocab_size=10000,
        n_embd=128,
        n_layer=3,
        n_head=4,
        block_size=32,
        batch_size=32,
        learning_rate=2e-4,
        max_iters=1000,
        description="Fast training optimized model"
    )
}

def get_experiment_config(name: str) -> ExperimentConfig:
    """Get experiment configuration by name"""
    if name not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment: {name}. Available: {list(EXPERIMENT_CONFIGS.keys())}")
    return EXPERIMENT_CONFIGS[name]

def list_experiments() -> List[str]:
    """List all available experiment names"""
    return list(EXPERIMENT_CONFIGS.keys())

def get_experiment_summary() -> Dict[str, str]:
    """Get summary of all experiments"""
    return {name: config.description for name, config in EXPERIMENT_CONFIGS.items()}

def save_experiment_results(experiment_name: str, results: Dict[str, Any], output_dir: str = "experiments"):
    """Save experiment results to JSON file"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{output_dir}/{experiment_name}_results.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")

def load_experiment_results(experiment_name: str, output_dir: str = "experiments") -> Dict[str, Any]:
    """Load experiment results from JSON file"""
    import os
    filename = f"{output_dir}/{experiment_name}_results.json"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Results file not found: {filename}")
    
    with open(filename, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    print("Available Experiments:")
    print("=" * 50)
    for name, config in EXPERIMENT_CONFIGS.items():
        print(f"{name:20} - {config.description}")
        print(f"{'':20}   Vocab: {config.vocab_size:,}, Emb: {config.n_embd}, Layers: {config.n_layer}, Heads: {config.n_head}")
        print()

