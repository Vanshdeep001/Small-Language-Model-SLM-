#!/usr/bin/env python3
"""
Simple Experiment Runner for SLM
Easy-to-use interface for running individual experiments
"""

import sys
import os
from experiment_config import EXPERIMENT_CONFIGS, list_experiments, get_experiment_summary
from run_experiment import run_experiment

def print_available_experiments():
    """Print all available experiments with descriptions"""
    print("\n" + "="*80)
    print("AVAILABLE EXPERIMENTS")
    print("="*80)
    
    categories = {
        "Baseline": ["baseline"],
        "Vocabulary Size": ["vocab_small", "vocab_medium"],
        "Embedding Dimensions": ["embd_small", "embd_large", "embd_xlarge"],
        "Architecture": ["arch_deep", "arch_wide"],
        "Training Data": ["data_small", "data_medium"],
        "Learning Rate": ["lr_high", "lr_low"],
        "Optimized": ["optimal_small", "optimal_fast"]
    }
    
    for category, experiments in categories.items():
        print(f"\n{category}:")
        print("-" * 40)
        for exp_name in experiments:
            if exp_name in EXPERIMENT_CONFIGS:
                config = EXPERIMENT_CONFIGS[exp_name]
                print(f"  {exp_name:20} - {config.description}")
                print(f"  {'':20}   Vocab: {config.vocab_size:,}, Emb: {config.n_embd}, Layers: {config.n_layer}")

def print_experiment_details(exp_name):
    """Print detailed information about a specific experiment"""
    if exp_name not in EXPERIMENT_CONFIGS:
        print(f"Unknown experiment: {exp_name}")
        return
    
    config = EXPERIMENT_CONFIGS[exp_name]
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT DETAILS: {exp_name}")
    print(f"{'='*60}")
    print(f"Description: {config.description}")
    print(f"\nConfiguration:")
    print(f"  Vocabulary Size: {config.vocab_size:,}")
    print(f"  Embedding Dimension: {config.n_embd}")
    print(f"  Number of Layers: {config.n_layer}")
    print(f"  Attention Heads: {config.n_head}")
    print(f"  Block Size: {config.block_size}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Max Iterations: {config.max_iters}")
    print(f"  Dropout: {config.dropout}")
    print(f"  Dataset Subset: {config.dataset_subset*100:.1f}%")
    
    # Estimate model size
    vocab_params = config.vocab_size * config.n_embd
    pos_params = config.block_size * config.n_embd
    attention_params = config.n_layer * (4 * config.n_embd * config.n_embd)  # Simplified
    mlp_params = config.n_layer * (8 * config.n_embd * config.n_embd)  # Simplified
    total_params = vocab_params + pos_params + attention_params + mlp_params
    
    print(f"\nEstimated Model Size:")
    print(f"  Total Parameters: ~{total_params:,}")
    print(f"  Memory (MB): ~{total_params * 4 / 1024 / 1024:.1f}")

def run_interactive_experiment():
    """Interactive experiment runner"""
    print("\n" + "="*60)
    print("SLM EXPERIMENT RUNNER")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. List all experiments")
        print("2. Show experiment details")
        print("3. Run experiment")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            print_available_experiments()
        
        elif choice == "2":
            print_available_experiments()
            exp_name = input("\nEnter experiment name: ").strip()
            print_experiment_details(exp_name)
        
        elif choice == "3":
            print_available_experiments()
            exp_name = input("\nEnter experiment name to run: ").strip()
            
            if exp_name not in EXPERIMENT_CONFIGS:
                print(f"Unknown experiment: {exp_name}")
                continue
            
            confirm = input(f"\nRun experiment '{exp_name}'? (y/n): ").strip().lower()
            if confirm == 'y':
                print(f"\nStarting experiment: {exp_name}")
                try:
                    results = run_experiment(exp_name)
                    if results:
                        print(f"\n✓ Experiment '{exp_name}' completed successfully!")
                    else:
                        print(f"\n✗ Experiment '{exp_name}' failed!")
                except Exception as e:
                    print(f"\n✗ Error running experiment '{exp_name}': {e}")
        
        elif choice == "4":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1-4.")

def main():
    """Main function"""
    if len(sys.argv) == 1:
        # Interactive mode
        run_interactive_experiment()
        return
    
    command = sys.argv[1].lower()
    
    if command == "list":
        print_available_experiments()
    
    elif command == "details":
        if len(sys.argv) < 3:
            print("Usage: python experiment_runner.py details <experiment_name>")
            print_available_experiments()
            return
        exp_name = sys.argv[2]
        print_experiment_details(exp_name)
    
    elif command == "run":
        if len(sys.argv) < 3:
            print("Usage: python experiment_runner.py run <experiment_name>")
            print_available_experiments()
            return
        exp_name = sys.argv[2]
        
        if exp_name not in EXPERIMENT_CONFIGS:
            print(f"Unknown experiment: {exp_name}")
            print_available_experiments()
            return
        
        print(f"Running experiment: {exp_name}")
        try:
            results = run_experiment(exp_name)
            if results:
                print(f"\n✓ Experiment '{exp_name}' completed successfully!")
            else:
                print(f"\n✗ Experiment '{exp_name}' failed!")
        except Exception as e:
            print(f"\n✗ Error running experiment '{exp_name}': {e}")
    
    else:
        print(f"Unknown command: {command}")
        print("\nUsage:")
        print("  python experiment_runner.py                    # Interactive mode")
        print("  python experiment_runner.py list              # List experiments")
        print("  python experiment_runner.py details <name>     # Show details")
        print("  python experiment_runner.py run <name>          # Run experiment")

if __name__ == "__main__":
    main()

