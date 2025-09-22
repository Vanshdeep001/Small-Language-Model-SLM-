#!/usr/bin/env python3
"""
Interactive Experiment Runner
Interactive command-line interface for running SLM experiments
"""

import os
import sys
import subprocess
from experiment_config import EXPERIMENT_CONFIGS, EXPERIMENT_CATEGORIES, get_experiment_config

def print_welcome():
    """Print welcome message"""
    print("🔬 SLM Experiment Runner")
    print("=" * 50)
    print("Interactive interface for running Small Language Model experiments")
    print()

def print_help():
    """Print help information"""
    print("\n📖 Available Commands:")
    print("  run <experiment_name>     - Run a specific experiment")
    print("  batch <category>          - Run all experiments in a category")
    print("  list [category]           - List available experiments")
    print("  details <experiment_name> - Show experiment details")
    print("  compare [category]        - Compare existing results")
    print("  help                      - Show this help")
    print("  quit                      - Exit the program")
    print()

def list_experiments(category=None):
    """List available experiments"""
    if category:
        if category not in EXPERIMENT_CATEGORIES:
            print(f"❌ Unknown category: {category}")
            print(f"Available categories: {', '.join(EXPERIMENT_CATEGORIES.keys())}")
            return
        
        experiments = EXPERIMENT_CATEGORIES[category]
        print(f"\n📊 {category.upper()} EXPERIMENTS:")
    else:
        experiments = list(EXPERIMENT_CONFIGS.keys())
        print(f"\n📋 ALL EXPERIMENTS ({len(experiments)} total):")
    
    print("-" * 60)
    for exp_name in experiments:
        config = EXPERIMENT_CONFIGS[exp_name]
        print(f"  • {exp_name:<20} - {config.description}")
    
    print()

def show_experiment_details(experiment_name):
    """Show detailed information about an experiment"""
    try:
        config = get_experiment_config(experiment_name)
        
        print(f"\n🔍 EXPERIMENT DETAILS: {experiment_name}")
        print("=" * 60)
        print(f"📝 Description: {config.description}")
        print(f"🔢 Vocabulary Size: {config.vocab_size:,}")
        print(f"📐 Embedding Dimension: {config.n_embd}")
        print(f"🏗️  Layers: {config.n_layer}")
        print(f"👁️  Attention Heads: {config.n_head}")
        print(f"📏 Block Size: {config.block_size}")
        print(f"📦 Batch Size: {config.batch_size}")
        print(f"📈 Learning Rate: {config.learning_rate}")
        print(f"🔄 Max Iterations: {config.max_iters:,}")
        print(f"📊 Dataset Fraction: {config.dataset_fraction*100:.1f}%")
        
        # Estimate parameters
        estimated_params = (
            config.vocab_size * config.n_embd +  # token embeddings
            config.block_size * config.n_embd +  # position embeddings
            config.n_layer * (
                4 * config.n_embd * config.n_embd +  # attention layers
                8 * config.n_embd * config.n_embd    # MLP layers
            ) +
            config.n_embd * config.vocab_size  # output head
        )
        print(f"🧮 Estimated Parameters: {estimated_params:,}")
        
        # Check if already exists
        model_path = f"best_model_{experiment_name}.pt"
        if os.path.exists(model_path):
            print(f"✅ Model already exists: {model_path}")
        else:
            print(f"⏳ Model not yet trained")
        
        print()
        
    except ValueError as e:
        print(f"❌ Error: {e}")

def run_experiment(experiment_name):
    """Run a single experiment"""
    try:
        config = get_experiment_config(experiment_name)
        
        print(f"\n🚀 Starting experiment: {experiment_name}")
        print(f"📝 {config.description}")
        print("=" * 60)
        
        # Check if model already exists
        model_path = f"best_model_{experiment_name}.pt"
        if os.path.exists(model_path):
            response = input(f"⚠️  Model {model_path} already exists. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("❌ Experiment cancelled")
                return
        
        # Run the experiment
        cmd = ["python", "run_experiment.py", experiment_name, "--verbose"]
        print(f"🔄 Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print(f"\n✅ Experiment '{experiment_name}' completed successfully!")
        else:
            print(f"\n❌ Experiment '{experiment_name}' failed!")
            
    except ValueError as e:
        print(f"❌ Error: {e}")

def run_batch_experiments(category):
    """Run batch experiments"""
    if category not in EXPERIMENT_CATEGORIES:
        print(f"❌ Unknown category: {category}")
        print(f"Available categories: {', '.join(EXPERIMENT_CATEGORIES.keys())}")
        return
    
    experiments = EXPERIMENT_CATEGORIES[category]
    print(f"\n🔬 BATCH EXPERIMENTS: {category.upper()}")
    print(f"📋 Will run {len(experiments)} experiments:")
    for exp_name in experiments:
        config = EXPERIMENT_CONFIGS[exp_name]
        print(f"  • {exp_name}: {config.description}")
    
    response = input(f"\n⚠️  This will run {len(experiments)} experiments. Continue? (y/N): ")
    if response.lower() != 'y':
        print("❌ Batch experiment cancelled")
        return
    
    print(f"\n🚀 Starting batch experiments...")
    cmd = ["python", "run_batch_experiments.py", category, "--verbose"]
    print(f"🔄 Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"\n✅ Batch experiments completed successfully!")
    else:
        print(f"\n❌ Batch experiments failed!")

def compare_results(category=None):
    """Compare existing results"""
    print(f"\n📊 COMPARING RESULTS")
    print("=" * 50)
    
    cmd = ["python", "run_batch_experiments.py", category or "all", "--compare-only"]
    print(f"🔄 Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"\n✅ Comparison completed!")
        print("📋 Check experiments/experiment_comparison.csv for detailed results")
        print("📊 Check experiments/experiment_comparison.png for visual comparison")
    else:
        print(f"\n❌ Comparison failed!")

def main():
    print_welcome()
    print_help()
    
    while True:
        try:
            command = input("🔬 experiment> ").strip().split()
            
            if not command:
                continue
            
            cmd = command[0].lower()
            
            if cmd == "quit" or cmd == "exit":
                print("👋 Goodbye!")
                break
            
            elif cmd == "help":
                print_help()
            
            elif cmd == "list":
                category = command[1] if len(command) > 1 else None
                list_experiments(category)
            
            elif cmd == "details":
                if len(command) < 2:
                    print("❌ Please specify an experiment name")
                    print("Usage: details <experiment_name>")
                    continue
                show_experiment_details(command[1])
            
            elif cmd == "run":
                if len(command) < 2:
                    print("❌ Please specify an experiment name")
                    print("Usage: run <experiment_name>")
                    continue
                run_experiment(command[1])
            
            elif cmd == "batch":
                if len(command) < 2:
                    print("❌ Please specify a category")
                    print("Usage: batch <category>")
                    print(f"Available categories: {', '.join(EXPERIMENT_CATEGORIES.keys())}")
                    continue
                run_batch_experiments(command[1])
            
            elif cmd == "compare":
                category = command[1] if len(command) > 1 else None
                compare_results(category)
            
            else:
                print(f"❌ Unknown command: {cmd}")
                print("Type 'help' for available commands")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"💥 Error: {e}")

if __name__ == "__main__":
    main()
