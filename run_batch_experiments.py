#!/usr/bin/env python3
"""
Batch Experiment Runner for SLM
Runs multiple experiments in sequence and compares results
"""

import os
import sys
import json
import time
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
from experiment_config import EXPERIMENT_CONFIGS, list_experiments, get_experiment_summary
from run_experiment import run_experiment

def run_batch_experiments(experiment_names: List[str] = None, 
                         skip_existing: bool = True,
                         max_experiments: int = None):
    """
    Run multiple experiments in batch
    
    Args:
        experiment_names: List of experiment names to run. If None, runs all experiments.
        skip_existing: Whether to skip experiments that already have results
        max_experiments: Maximum number of experiments to run (for testing)
    """
    
    if experiment_names is None:
        experiment_names = list_experiments()
    
    if max_experiments:
        experiment_names = experiment_names[:max_experiments]
    
    print(f"\n{'='*80}")
    print(f"BATCH EXPERIMENT RUNNER")
    print(f"{'='*80}")
    print(f"Total experiments to run: {len(experiment_names)}")
    print(f"Skip existing results: {skip_existing}")
    print(f"Experiments: {', '.join(experiment_names)}")
    print(f"{'='*80}")
    
    # Create experiments directory
    os.makedirs("experiments", exist_ok=True)
    
    results = {}
    failed_experiments = []
    
    for i, exp_name in enumerate(experiment_names, 1):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {i}/{len(experiment_names)}: {exp_name}")
        print(f"{'='*60}")
        
        # Check if results already exist
        results_file = f"experiments/{exp_name}_results.json"
        if skip_existing and os.path.exists(results_file):
            print(f"Results already exist for {exp_name}, skipping...")
            try:
                with open(results_file, 'r') as f:
                    results[exp_name] = json.load(f)
                print(f"Loaded existing results for {exp_name}")
                continue
            except Exception as e:
                print(f"Error loading existing results: {e}")
        
        # Run experiment
        try:
            start_time = time.time()
            exp_results = run_experiment(exp_name)
            end_time = time.time()
            
            if exp_results:
                results[exp_name] = exp_results
                print(f"✓ Experiment {exp_name} completed successfully in {end_time - start_time:.2f}s")
            else:
                failed_experiments.append(exp_name)
                print(f"✗ Experiment {exp_name} failed")
                
        except Exception as e:
            failed_experiments.append(exp_name)
            print(f"✗ Experiment {exp_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print(f"BATCH EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {len(experiment_names)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(failed_experiments)}")
    
    if failed_experiments:
        print(f"Failed experiments: {', '.join(failed_experiments)}")
    
    # Save batch results
    batch_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_experiments": len(experiment_names),
        "successful_experiments": len(results),
        "failed_experiments": len(failed_experiments),
        "failed_experiment_names": failed_experiments,
        "experiment_results": results
    }
    
    with open("experiments/batch_results.json", 'w') as f:
        json.dump(batch_results, f, indent=2)
    
    print(f"\nBatch results saved to: experiments/batch_results.json")
    
    return results, failed_experiments

def compare_experiments(results: Dict[str, Any] = None):
    """
    Compare results from multiple experiments
    """
    
    if results is None:
        # Load results from file
        try:
            with open("experiments/batch_results.json", 'r') as f:
                batch_data = json.load(f)
                results = batch_data["experiment_results"]
        except FileNotFoundError:
            print("No batch results found. Please run experiments first.")
            return
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPARISON")
    print(f"{'='*80}")
    
    # Create comparison table
    comparison_data = []
    
    for exp_name, exp_results in results.items():
        config = exp_results["config"]
        model_info = exp_results["model_info"]
        training_results = exp_results["training_results"]
        
        comparison_data.append({
            "Experiment": exp_name,
            "Description": exp_results["description"],
            "Vocab Size": f"{config['vocab_size']:,}",
            "Embedding Dim": config['n_embd'],
            "Layers": config['n_layer'],
            "Heads": config['n_head'],
            "Parameters": f"{model_info['total_parameters']:,}",
            "Training Time (s)": f"{model_info['training_time_seconds']:.1f}",
            "Final Train Loss": f"{training_results['final_train_loss']:.4f}" if training_results['final_train_loss'] else "N/A",
            "Final Val Loss": f"{training_results['final_val_loss']:.4f}" if training_results['final_val_loss'] else "N/A",
            "Best Val Loss": f"{training_results['best_val_loss']:.4f}",
            "Data Subset": f"{config['dataset_subset']*100:.1f}%"
        })
    
    # Create DataFrame for better formatting
    df = pd.DataFrame(comparison_data)
    
    # Print comparison table
    print("\nEXPERIMENT COMPARISON TABLE:")
    print("-" * 120)
    print(df.to_string(index=False))
    
    # Save comparison table
    df.to_csv("experiments/experiment_comparison.csv", index=False)
    print(f"\nComparison table saved to: experiments/experiment_comparison.csv")
    
    # Create visualizations
    create_comparison_plots(results)
    
    return df

def create_comparison_plots(results: Dict[str, Any]):
    """
    Create comparison plots for experiments
    """
    
    print("\nCreating comparison plots...")
    
    # Extract data for plotting
    exp_names = list(results.keys())
    vocab_sizes = [results[exp]["config"]["vocab_size"] for exp in exp_names]
    embedding_dims = [results[exp]["config"]["n_embd"] for exp in exp_names]
    parameters = [results[exp]["model_info"]["total_parameters"] for exp in exp_names]
    training_times = [results[exp]["model_info"]["training_time_seconds"] for exp in exp_names]
    final_val_losses = [results[exp]["training_results"]["final_val_loss"] for exp in exp_names if results[exp]["training_results"]["final_val_loss"]]
    best_val_losses = [results[exp]["training_results"]["best_val_loss"] for exp in exp_names]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SLM Experiment Comparison', fontsize=16)
    
    # Plot 1: Parameters vs Validation Loss
    axes[0, 0].scatter(parameters, best_val_losses, alpha=0.7, s=100)
    axes[0, 0].set_xlabel('Total Parameters')
    axes[0, 0].set_ylabel('Best Validation Loss')
    axes[0, 0].set_title('Parameters vs Validation Loss')
    axes[0, 0].grid(True, alpha=0.3)
    for i, exp in enumerate(exp_names):
        axes[0, 0].annotate(exp, (parameters[i], best_val_losses[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 2: Training Time vs Validation Loss
    axes[0, 1].scatter(training_times, best_val_losses, alpha=0.7, s=100, color='orange')
    axes[0, 1].set_xlabel('Training Time (seconds)')
    axes[0, 1].set_ylabel('Best Validation Loss')
    axes[0, 1].set_title('Training Time vs Validation Loss')
    axes[0, 1].grid(True, alpha=0.3)
    for i, exp in enumerate(exp_names):
        axes[0, 1].annotate(exp, (training_times[i], best_val_losses[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 3: Vocabulary Size vs Validation Loss
    axes[0, 2].scatter(vocab_sizes, best_val_losses, alpha=0.7, s=100, color='green')
    axes[0, 2].set_xlabel('Vocabulary Size')
    axes[0, 2].set_ylabel('Best Validation Loss')
    axes[0, 2].set_title('Vocabulary Size vs Validation Loss')
    axes[0, 2].grid(True, alpha=0.3)
    for i, exp in enumerate(exp_names):
        axes[0, 2].annotate(exp, (vocab_sizes[i], best_val_losses[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 4: Embedding Dimension vs Validation Loss
    axes[1, 0].scatter(embedding_dims, best_val_losses, alpha=0.7, s=100, color='red')
    axes[1, 0].set_xlabel('Embedding Dimension')
    axes[1, 0].set_ylabel('Best Validation Loss')
    axes[1, 0].set_title('Embedding Dimension vs Validation Loss')
    axes[1, 0].grid(True, alpha=0.3)
    for i, exp in enumerate(exp_names):
        axes[1, 0].annotate(exp, (embedding_dims[i], best_val_losses[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 5: Training Loss Curves
    axes[1, 1].set_title('Training Loss Curves')
    axes[1, 1].set_xlabel('Training Steps')
    axes[1, 1].set_ylabel('Loss')
    for exp_name, exp_results in results.items():
        train_losses = exp_results["training_results"]["train_losses"]
        val_losses = exp_results["training_results"]["val_losses"]
        if train_losses and val_losses:
            steps = list(range(0, len(train_losses) * 50, 50))  # Assuming eval every 50 steps
            axes[1, 1].plot(steps, train_losses, label=f'{exp_name} (train)', alpha=0.7)
            axes[1, 1].plot(steps, val_losses, label=f'{exp_name} (val)', linestyle='--', alpha=0.7)
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Model Size Comparison
    axes[1, 2].bar(range(len(exp_names)), parameters, alpha=0.7, color='purple')
    axes[1, 2].set_xlabel('Experiments')
    axes[1, 2].set_ylabel('Total Parameters')
    axes[1, 2].set_title('Model Size Comparison')
    axes[1, 2].set_xticks(range(len(exp_names)))
    axes[1, 2].set_xticklabels(exp_names, rotation=45, ha='right')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("experiments/experiment_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comparison plots saved to: experiments/experiment_comparison.png")

def run_quick_experiments():
    """
    Run a subset of quick experiments for testing
    """
    quick_experiments = [
        "baseline",
        "vocab_small", 
        "embd_small",
        "data_small",
        "optimal_fast"
    ]
    
    print("Running quick experiments for testing...")
    results, failed = run_batch_experiments(quick_experiments, max_experiments=5)
    
    if results:
        compare_experiments(results)
    
    return results, failed

def run_vocabulary_experiments():
    """
    Run experiments focused on vocabulary size variations
    """
    vocab_experiments = [
        "baseline",
        "vocab_small",
        "vocab_medium"
    ]
    
    print("Running vocabulary size experiments...")
    results, failed = run_batch_experiments(vocab_experiments)
    
    if results:
        compare_experiments(results)
    
    return results, failed

def run_embedding_experiments():
    """
    Run experiments focused on embedding dimension variations
    """
    embedding_experiments = [
        "baseline",
        "embd_small",
        "embd_large",
        "embd_xlarge"
    ]
    
    print("Running embedding dimension experiments...")
    results, failed = run_batch_experiments(embedding_experiments)
    
    if results:
        compare_experiments(results)
    
    return results, failed

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python run_batch_experiments.py <command>")
        print("\nAvailable commands:")
        print("  quick          - Run quick experiments for testing")
        print("  vocab          - Run vocabulary size experiments")
        print("  embedding      - Run embedding dimension experiments")
        print("  all            - Run all experiments")
        print("  compare        - Compare existing results")
        print("  list           - List available experiments")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "quick":
        run_quick_experiments()
    elif command == "vocab":
        run_vocabulary_experiments()
    elif command == "embedding":
        run_embedding_experiments()
    elif command == "all":
        results, failed = run_batch_experiments()
        if results:
            compare_experiments(results)
    elif command == "compare":
        compare_experiments()
    elif command == "list":
        print("Available experiments:")
        summary = get_experiment_summary()
        for name, desc in summary.items():
            print(f"  {name:20} - {desc}")
    else:
        print(f"Unknown command: {command}")
        print("Available commands: quick, vocab, embedding, all, compare, list")

