#!/usr/bin/env python3
"""
Experiment Configuration for SLM
Defines all experiment variations for systematic testing
"""

from dataclasses import dataclass
from typing import Dict, Any

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
    dataset_fraction: float = 1.0  # Fraction of dataset to use (0.1 = 10%, 1.0 = 100%)
    description: str = ""
    
    def __post_init__(self):
        # Validate configuration
        assert self.n_embd % self.n_head == 0, f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        assert 0 < self.dataset_fraction <= 1.0, f"dataset_fraction must be between 0 and 1, got {self.dataset_fraction}"

# Baseline Configuration (Current Working Model)
BASELINE = ExperimentConfig(
    name="baseline",
    vocab_size=50257,
    n_embd=256,
    n_layer=4,
    n_head=4,
    block_size=64,
    batch_size=32,
    learning_rate=1e-4,
    max_iters=2000,
    dataset_fraction=1.0,
    description="Current working configuration"
)

# Vocabulary Size Experiments
VOCAB_EXPERIMENTS = {
    "vocab_small": ExperimentConfig(
        name="vocab_small",
        vocab_size=10000,
        n_embd=256,
        n_layer=4,
        n_head=4,
        block_size=64,
        batch_size=32,
        learning_rate=1e-4,
        max_iters=2000,
        dataset_fraction=1.0,
        description="Small vocabulary (10K tokens)"
    ),
    
    "vocab_medium": ExperimentConfig(
        name="vocab_medium",
        vocab_size=25000,
        n_embd=256,
        n_layer=4,
        n_head=4,
        block_size=64,
        batch_size=32,
        learning_rate=1e-4,
        max_iters=2000,
        dataset_fraction=1.0,
        description="Medium vocabulary (25K tokens)"
    ),
    
    "vocab_large": ExperimentConfig(
        name="vocab_large",
        vocab_size=50257,
        n_embd=256,
        n_layer=4,
        n_head=4,
        block_size=64,
        batch_size=32,
        learning_rate=1e-4,
        max_iters=2000,
        dataset_fraction=1.0,
        description="Large vocabulary (50K tokens) - same as baseline"
    )
}

# Embedding Dimension Experiments
EMBEDDING_EXPERIMENTS = {
    "embd_small": ExperimentConfig(
        name="embd_small",
        vocab_size=50257,
        n_embd=128,
        n_layer=4,
        n_head=4,
        block_size=64,
        batch_size=32,
        learning_rate=1e-4,
        max_iters=2000,
        dataset_fraction=1.0,
        description="Small embeddings (128 dimensions)"
    ),
    
    "embd_medium": ExperimentConfig(
        name="embd_medium",
        vocab_size=50257,
        n_embd=256,
        n_layer=4,
        n_head=4,
        block_size=64,
        batch_size=32,
        learning_rate=1e-4,
        max_iters=2000,
        dataset_fraction=1.0,
        description="Medium embeddings (256 dimensions) - same as baseline"
    ),
    
    "embd_large": ExperimentConfig(
        name="embd_large",
        vocab_size=50257,
        n_embd=512,
        n_layer=4,
        n_head=8,
        block_size=64,
        batch_size=32,
        learning_rate=1e-4,
        max_iters=2000,
        dataset_fraction=1.0,
        description="Large embeddings (512 dimensions)"
    ),
    
    "embd_xlarge": ExperimentConfig(
        name="embd_xlarge",
        vocab_size=50257,
        n_embd=768,
        n_layer=4,
        n_head=8,
        block_size=64,
        batch_size=32,
        learning_rate=1e-4,
        max_iters=2000,
        dataset_fraction=1.0,
        description="Extra large embeddings (768 dimensions)"
    )
}

# Architecture Experiments
ARCHITECTURE_EXPERIMENTS = {
    "arch_shallow": ExperimentConfig(
        name="arch_shallow",
        vocab_size=50257,
        n_embd=256,
        n_layer=2,
        n_head=4,
        block_size=64,
        batch_size=32,
        learning_rate=1e-4,
        max_iters=2000,
        dataset_fraction=1.0,
        description="Shallow model (2 layers)"
    ),
    
    "arch_deep": ExperimentConfig(
        name="arch_deep",
        vocab_size=50257,
        n_embd=256,
        n_layer=8,
        n_head=4,
        block_size=64,
        batch_size=32,
        learning_rate=1e-4,
        max_iters=2000,
        dataset_fraction=1.0,
        description="Deep model (8 layers)"
    ),
    
    "arch_wide": ExperimentConfig(
        name="arch_wide",
        vocab_size=50257,
        n_embd=256,
        n_layer=4,
        n_head=8,
        block_size=64,
        batch_size=32,
        learning_rate=1e-4,
        max_iters=2000,
        dataset_fraction=1.0,
        description="Wide attention (8 heads)"
    ),
    
    "arch_narrow": ExperimentConfig(
        name="arch_narrow",
        vocab_size=50257,
        n_embd=256,
        n_layer=4,
        n_head=2,
        block_size=64,
        batch_size=32,
        learning_rate=1e-4,
        max_iters=2000,
        dataset_fraction=1.0,
        description="Narrow attention (2 heads)"
    )
}

# Learning Rate Experiments
LEARNING_RATE_EXPERIMENTS = {
    "lr_low": ExperimentConfig(
        name="lr_low",
        vocab_size=50257,
        n_embd=256,
        n_layer=4,
        n_head=4,
        block_size=64,
        batch_size=32,
        learning_rate=5e-5,
        max_iters=2000,
        dataset_fraction=1.0,
        description="Low learning rate (5e-5)"
    ),
    
    "lr_high": ExperimentConfig(
        name="lr_high",
        vocab_size=50257,
        n_embd=256,
        n_layer=4,
        n_head=4,
        block_size=64,
        batch_size=32,
        learning_rate=5e-4,
        max_iters=2000,
        dataset_fraction=1.0,
        description="High learning rate (5e-4)"
    ),
    
    "lr_very_high": ExperimentConfig(
        name="lr_very_high",
        vocab_size=50257,
        n_embd=256,
        n_layer=4,
        n_head=4,
        block_size=64,
        batch_size=32,
        learning_rate=1e-3,
        max_iters=2000,
        dataset_fraction=1.0,
        description="Very high learning rate (1e-3)"
    )
}

# Dataset Size Experiments
DATASET_SIZE_EXPERIMENTS = {
    "data_tiny": ExperimentConfig(
        name="data_tiny",
        vocab_size=50257,
        n_embd=256,
        n_layer=4,
        n_head=4,
        block_size=64,
        batch_size=32,
        learning_rate=1e-4,
        max_iters=2000,
        dataset_fraction=0.1,
        description="Tiny dataset (10% of data)"
    ),
    
    "data_small": ExperimentConfig(
        name="data_small",
        vocab_size=50257,
        n_embd=256,
        n_layer=4,
        n_head=4,
        block_size=64,
        batch_size=32,
        learning_rate=1e-4,
        max_iters=2000,
        dataset_fraction=0.25,
        description="Small dataset (25% of data)"
    ),
    
    "data_medium": ExperimentConfig(
        name="data_medium",
        vocab_size=50257,
        n_embd=256,
        n_layer=4,
        n_head=4,
        block_size=64,
        batch_size=32,
        learning_rate=1e-4,
        max_iters=2000,
        dataset_fraction=0.5,
        description="Medium dataset (50% of data)"
    ),
    
    "data_large": ExperimentConfig(
        name="data_large",
        vocab_size=50257,
        n_embd=256,
        n_layer=4,
        n_head=4,
        block_size=64,
        batch_size=32,
        learning_rate=1e-4,
        max_iters=2000,
        dataset_fraction=1.0,
        description="Large dataset (100% of data) - same as baseline"
    )
}

# Optimized Configurations
OPTIMIZED_EXPERIMENTS = {
    "optimal_small": ExperimentConfig(
        name="optimal_small",
        vocab_size=10000,
        n_embd=128,
        n_layer=2,
        n_head=4,
        block_size=64,
        batch_size=32,
        learning_rate=2e-4,
        max_iters=1500,
        dataset_fraction=0.5,
        description="Optimized small model for speed"
    ),
    
    "optimal_fast": ExperimentConfig(
        name="optimal_fast",
        vocab_size=25000,
        n_embd=256,
        n_layer=3,
        n_head=4,
        block_size=64,
        batch_size=32,
        learning_rate=1e-4,
        max_iters=1000,
        dataset_fraction=0.75,
        description="Optimized for fast training"
    ),
    
    "optimal_quality": ExperimentConfig(
        name="optimal_quality",
        vocab_size=50257,
        n_embd=512,
        n_layer=6,
        n_head=8,
        block_size=64,
        batch_size=32,
        learning_rate=5e-5,
        max_iters=3000,
        dataset_fraction=1.0,
        description="Optimized for quality"
    )
}

# Combine all experiments
EXPERIMENT_CONFIGS = {
    "baseline": BASELINE,
    **VOCAB_EXPERIMENTS,
    **EMBEDDING_EXPERIMENTS,
    **ARCHITECTURE_EXPERIMENTS,
    **LEARNING_RATE_EXPERIMENTS,
    **DATASET_SIZE_EXPERIMENTS,
    **OPTIMIZED_EXPERIMENTS
}

# Experiment categories for batch running
EXPERIMENT_CATEGORIES = {
    "vocab": list(VOCAB_EXPERIMENTS.keys()),
    "embedding": list(EMBEDDING_EXPERIMENTS.keys()),
    "architecture": list(ARCHITECTURE_EXPERIMENTS.keys()),
    "learning_rate": list(LEARNING_RATE_EXPERIMENTS.keys()),
    "dataset_size": list(DATASET_SIZE_EXPERIMENTS.keys()),
    "optimized": list(OPTIMIZED_EXPERIMENTS.keys()),
    "quick": ["baseline", "vocab_small", "embd_small", "arch_shallow", "data_tiny"],
    "all": list(EXPERIMENT_CONFIGS.keys())
}

def get_experiment_config(name: str) -> ExperimentConfig:
    """Get experiment configuration by name"""
    if name not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment: {name}. Available: {list(EXPERIMENT_CONFIGS.keys())}")
    return EXPERIMENT_CONFIGS[name]

def list_experiments(category: str = None) -> Dict[str, ExperimentConfig]:
    """List experiments, optionally filtered by category"""
    if category is None:
        return EXPERIMENT_CONFIGS
    
    if category not in EXPERIMENT_CATEGORIES:
        raise ValueError(f"Unknown category: {category}. Available: {list(EXPERIMENT_CATEGORIES.keys())}")
    
    return {name: EXPERIMENT_CONFIGS[name] for name in EXPERIMENT_CATEGORIES[category]}

def print_experiment_summary():
    """Print a summary of all available experiments"""
    print("🔬 Available Experiments:")
    print("=" * 50)
    
    for category, experiments in EXPERIMENT_CATEGORIES.items():
        if category in ["quick", "all"]:
            continue
        print(f"\n📊 {category.upper()} EXPERIMENTS:")
        for exp_name in experiments:
            config = EXPERIMENT_CONFIGS[exp_name]
            print(f"  • {exp_name}: {config.description}")
    
    print(f"\n🚀 QUICK EXPERIMENTS: {', '.join(EXPERIMENT_CATEGORIES['quick'])}")
    print(f"📋 ALL EXPERIMENTS: {len(EXPERIMENT_CONFIGS)} total")

if __name__ == "__main__":
    print_experiment_summary()
