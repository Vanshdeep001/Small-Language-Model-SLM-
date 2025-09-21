#!/usr/bin/env python3
"""
Demo Script for SLM Experiments
Demonstrates how to run key experiments and compare results
"""

import os
import sys
import time
from experiment_config import get_experiment_config
from run_experiment import run_experiment
from run_batch_experiments import compare_experiments

def run_demo_experiments():
    """
    Run a demonstration of key experiments
    """
    print("\n" + "="*80)
    print("SLM EXPERIMENT DEMONSTRATION")
    print("="*80)
    print("This demo will run 3 key experiments:")
    print("1. Baseline - Current best configuration")
    print("2. Small Vocabulary - Reduced vocabulary size")
    print("3. Small Embeddings - Reduced embedding dimension")
    print("\nEach experiment will take 10-30 minutes to complete.")
    print("="*80)
    
    # Demo experiments
    demo_experiments = [
        "baseline",
        "vocab_small", 
        "embd_small"
    ]
    
    results = {}
    total_start_time = time.time()
    
    for i, exp_name in enumerate(demo_experiments, 1):
        print(f"\n{'='*60}")
        print(f"DEMO EXPERIMENT {i}/3: {exp_name}")
        print(f"{'='*60}")
        
        # Show experiment details
        config = get_experiment_config(exp_name)
        print(f"Description: {config.description}")
        print(f"Configuration:")
        print(f"  Vocabulary Size: {config.vocab_size:,}")
        print(f"  Embedding Dimension: {config.n_embd}")
        print(f"  Layers: {config.n_layer}")
        print(f"  Attention Heads: {config.n_head}")
        print(f"  Max Iterations: {config.max_iters}")
        
        # Run experiment
        try:
            start_time = time.time()
            exp_results = run_experiment(exp_name)
            end_time = time.time()
            
            if exp_results:
                results[exp_name] = exp_results
                print(f"\n✓ Experiment {exp_name} completed successfully!")
                print(f"  Training time: {end_time - start_time:.1f} seconds")
                print(f"  Best validation loss: {exp_results['training_results']['best_val_loss']:.4f}")
                print(f"  Total parameters: {exp_results['model_info']['total_parameters']:,}")
            else:
                print(f"\n✗ Experiment {exp_name} failed!")
                
        except Exception as e:
            print(f"\n✗ Experiment {exp_name} failed with error: {e}")
    
    total_time = time.time() - total_start_time
    
    # Summary
    print(f"\n{'='*80}")
    print(f"DEMO EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Successful experiments: {len(results)}")
    print(f"Failed experiments: {len(demo_experiments) - len(results)}")
    
    if results:
        print(f"\nResults Summary:")
        print("-" * 60)
        for exp_name, exp_results in results.items():
            config = exp_results['config']
            model_info = exp_results['model_info']
            training_results = exp_results['training_results']
            
            print(f"{exp_name:15} | Params: {model_info['total_parameters']:8,} | "
                  f"Val Loss: {training_results['best_val_loss']:6.4f} | "
                  f"Time: {model_info['training_time_seconds']:6.1f}s")
        
        # Compare results
        print(f"\n{'='*60}")
        print("COMPARING RESULTS")
        print(f"{'='*60}")
        compare_experiments(results)
        
        print(f"\n{'='*80}")
        print("DEMO COMPLETED!")
        print("="*80)
        print("Check the following files for detailed results:")
        print("- experiments/batch_results.json")
        print("- experiments/experiment_comparison.csv")
        print("- experiments/experiment_comparison.png")
        print("="*80)
    
    return results

def run_single_demo():
    """
    Run a single quick experiment for demonstration
    """
    print("\n" + "="*60)
    print("SINGLE EXPERIMENT DEMO")
    print("="*60)
    print("Running 'optimal_fast' experiment - designed for quick training")
    print("This should complete in 5-15 minutes")
    print("="*60)
    
    try:
        results = run_experiment("optimal_fast")
        if results:
            print(f"\n✓ Demo experiment completed successfully!")
            print(f"Best validation loss: {results['training_results']['best_val_loss']:.4f}")
            print(f"Total parameters: {results['model_info']['total_parameters']:,}")
            print(f"Training time: {results['model_info']['training_time_seconds']:.1f} seconds")
            
            # Show generated samples
            print(f"\nGenerated samples:")
            for i, sample in enumerate(results['generated_samples'], 1):
                print(f"{i}. {sample[:100]}...")
        else:
            print(f"\n✗ Demo experiment failed!")
    except Exception as e:
        print(f"\n✗ Error running demo experiment: {e}")

def show_experiment_info():
    """
    Show information about available experiments
    """
    print("\n" + "="*80)
    print("AVAILABLE EXPERIMENTS")
    print("="*80)
    
    from experiment_config import EXPERIMENT_CONFIGS
    
    categories = {
        "Quick Experiments": ["baseline", "vocab_small", "embd_small", "optimal_fast"],
        "Vocabulary Experiments": ["vocab_small", "vocab_medium"],
        "Embedding Experiments": ["embd_small", "embd_large", "embd_xlarge"],
        "Architecture Experiments": ["arch_deep", "arch_wide"],
        "Data Experiments": ["data_small", "data_medium"],
        "Learning Rate Experiments": ["lr_high", "lr_low"],
        "Optimized Configurations": ["optimal_small", "optimal_fast"]
    }
    
    for category, experiments in categories.items():
        print(f"\n{category}:")
        print("-" * 40)
        for exp_name in experiments:
            if exp_name in EXPERIMENT_CONFIGS:
                config = EXPERIMENT_CONFIGS[exp_name]
                print(f"  {exp_name:20} - {config.description}")
                print(f"  {'':20}   Vocab: {config.vocab_size:,}, Emb: {config.n_embd}, "
                      f"Layers: {config.n_layer}, Iters: {config.max_iters}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demo_experiments.py <command>")
        print("\nAvailable commands:")
        print("  demo     - Run 3 key experiments (30-60 minutes)")
        print("  single   - Run single quick experiment (5-15 minutes)")
        print("  info     - Show information about all experiments")
        print("\nExample:")
        print("  python demo_experiments.py single")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "demo":
        run_demo_experiments()
    elif command == "single":
        run_single_demo()
    elif command == "info":
        show_experiment_info()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: demo, single, info")

