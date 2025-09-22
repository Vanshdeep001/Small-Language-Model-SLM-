#!/usr/bin/env python3
"""
Batch Experiment Runner
Runs multiple experiments and compares results
"""

import os
import json
import time
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import argparse

from experiment_config import EXPERIMENT_CATEGORIES, get_experiment_config

def run_experiment(experiment_name: str, verbose: bool = False) -> bool:
    """Run a single experiment"""
    print(f"\n🚀 Running experiment: {experiment_name}")
    print("=" * 50)
    
    try:
        cmd = ["python", "run_experiment.py", experiment_name]
        if verbose:
            cmd.append("--verbose")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f"✅ {experiment_name} completed successfully")
            return True
            else:
            print(f"❌ {experiment_name} failed:")
            print(result.stderr)
            return False
                
    except subprocess.TimeoutExpired:
        print(f"⏰ {experiment_name} timed out after 1 hour")
        return False
        except Exception as e:
        print(f"💥 {experiment_name} crashed: {e}")
        return False

def load_experiment_results(experiment_name: str) -> Dict:
    """Load results from a completed experiment"""
    results_path = f"experiments/{experiment_name}_results.json"
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None

def run_batch_experiments(experiment_names: List[str], verbose: bool = False) -> Dict[str, bool]:
    """Run a batch of experiments"""
    results = {}
    start_time = time.time()
    
    print(f"🔬 Starting batch of {len(experiment_names)} experiments...")
    print(f"📋 Experiments: {', '.join(experiment_names)}")
    
    for i, exp_name in enumerate(experiment_names, 1):
        print(f"\n📊 Progress: {i}/{len(experiment_names)}")
        success = run_experiment(exp_name, verbose)
        results[exp_name] = success
        
        if not success:
            print(f"⚠️  Skipping remaining experiments due to failure")
            break
    
    total_time = time.time() - start_time
    successful = sum(results.values())
    
    print(f"\n🎯 Batch completed!")
    print(f"✅ Successful: {successful}/{len(experiment_names)}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    
    return results

def create_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Create a comparison table from experiment results"""
    data = []
    
    for exp_name, result in results.items():
        if result is None:
            continue
            
        config = result['config']
        res = result['results']
        
        data.append({
            'Experiment': exp_name,
            'Description': result['description'],
            'Vocab Size': config['vocab_size'],
            'Embedding Dim': config['n_embd'],
            'Layers': config['n_layer'],
            'Heads': config['n_head'],
            'Learning Rate': config['learning_rate'],
            'Dataset %': config['dataset_fraction'] * 100,
            'Parameters': res['total_parameters'],
            'Training Time (min)': res['training_time_seconds'] / 60,
            'Final Train Loss': res['final_train_loss'],
            'Final Val Loss': res['final_val_loss'],
            'Best Val Loss': res['best_val_loss']
        })
    
    return pd.DataFrame(data)

def create_comparison_plots(df: pd.DataFrame, output_dir: str = "experiments"):
    """Create comparison plots"""
    if df.empty:
        print("❌ No data to plot")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SLM Experiment Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Parameters vs Validation Loss
    axes[0, 0].scatter(df['Parameters'], df['Best Val Loss'], alpha=0.7, s=100)
    axes[0, 0].set_xlabel('Total Parameters')
    axes[0, 0].set_ylabel('Best Validation Loss')
    axes[0, 0].set_title('Model Size vs Performance')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add experiment labels
    for i, row in df.iterrows():
        axes[0, 0].annotate(row['Experiment'], (row['Parameters'], row['Best Val Loss']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 2: Training Time vs Performance
    axes[0, 1].scatter(df['Training Time (min)'], df['Best Val Loss'], alpha=0.7, s=100)
    axes[0, 1].set_xlabel('Training Time (minutes)')
    axes[0, 1].set_ylabel('Best Validation Loss')
    axes[0, 1].set_title('Training Efficiency vs Performance')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate vs Performance
    axes[0, 2].scatter(df['Learning Rate'], df['Best Val Loss'], alpha=0.7, s=100)
    axes[0, 2].set_xlabel('Learning Rate')
    axes[0, 2].set_ylabel('Best Validation Loss')
    axes[0, 2].set_title('Learning Rate vs Performance')
    axes[0, 2].set_xscale('log')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Embedding Dimension vs Performance
    axes[1, 0].scatter(df['Embedding Dim'], df['Best Val Loss'], alpha=0.7, s=100)
    axes[1, 0].set_xlabel('Embedding Dimension')
    axes[1, 0].set_ylabel('Best Validation Loss')
    axes[1, 0].set_title('Embedding Size vs Performance')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Layers vs Performance
    axes[1, 1].scatter(df['Layers'], df['Best Val Loss'], alpha=0.7, s=100)
    axes[1, 1].set_xlabel('Number of Layers')
    axes[1, 1].set_ylabel('Best Validation Loss')
    axes[1, 1].set_title('Model Depth vs Performance')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Dataset Size vs Performance
    axes[1, 2].scatter(df['Dataset %'], df['Best Val Loss'], alpha=0.7, s=100)
    axes[1, 2].set_xlabel('Dataset Percentage')
    axes[1, 2].set_ylabel('Best Validation Loss')
    axes[1, 2].set_title('Data Size vs Performance')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "experiment_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparison plots saved to: {plot_path}")

def save_batch_results(results: Dict[str, Dict], output_dir: str = "experiments"):
    """Save batch results to JSON"""
    batch_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_experiments": len(results),
        "successful_experiments": len([r for r in results.values() if r is not None]),
        "experiments": results
    }
    
    batch_path = os.path.join(output_dir, "batch_results.json")
    with open(batch_path, 'w') as f:
        json.dump(batch_results, f, indent=2)
    
    print(f"💾 Batch results saved to: {batch_path}")

def main():
    parser = argparse.ArgumentParser(description='Run batch SLM experiments')
    parser.add_argument('category', help='Experiment category to run', 
                       choices=list(EXPERIMENT_CATEGORIES.keys()) + ['custom'])
    parser.add_argument('--experiments', nargs='+', help='Custom experiment names (use with category=custom)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--compare-only', action='store_true', help='Only create comparison from existing results')
    
    args = parser.parse_args()
    
    # Create experiments directory
    os.makedirs("experiments", exist_ok=True)
    
    if args.compare_only:
        # Load existing results and create comparison
        print("📊 Creating comparison from existing results...")
        results = {}
        for exp_name in EXPERIMENT_CATEGORIES.get(args.category, []):
            result = load_experiment_results(exp_name)
            if result:
                results[exp_name] = result
        
        if not results:
            print("❌ No existing results found")
            return
        
        df = create_comparison_table(results)
        df.to_csv("experiments/experiment_comparison.csv", index=False)
        create_comparison_plots(df)
        print(f"📋 Comparison table saved to: experiments/experiment_comparison.csv")
        
    else:
        # Run experiments
        if args.category == 'custom':
            if not args.experiments:
                print("❌ Please provide experiment names with --experiments")
                return
            experiment_names = args.experiments
        else:
            experiment_names = EXPERIMENT_CATEGORIES[args.category]
        
        # Run batch experiments
        success_results = run_batch_experiments(experiment_names, args.verbose)
        
        # Load all results
        all_results = {}
        for exp_name in experiment_names:
            result = load_experiment_results(exp_name)
            if result:
                all_results[exp_name] = result
        
        # Save batch results
        save_batch_results(all_results)
        
        # Create comparison
        if all_results:
            df = create_comparison_table(all_results)
            df.to_csv("experiments/experiment_comparison.csv", index=False)
            create_comparison_plots(df)
            print(f"📋 Comparison table saved to: experiments/experiment_comparison.csv")
            
            # Print summary
            print(f"\n🏆 BEST PERFORMING EXPERIMENTS:")
            print("=" * 50)
            
            # Sort by validation loss
            best_loss = df.loc[df['Best Val Loss'].idxmin()]
            print(f"🥇 Best Validation Loss: {best_loss['Experiment']} ({best_loss['Best Val Loss']:.4f})")
            
            # Sort by training efficiency
            df['Efficiency'] = df['Best Val Loss'] / df['Training Time (min)']
            most_efficient = df.loc[df['Efficiency'].idxmax()]
            print(f"⚡ Most Efficient: {most_efficient['Experiment']} ({most_efficient['Efficiency']:.4f} loss/min)")
            
            # Sort by parameter efficiency
            df['Param Efficiency'] = df['Best Val Loss'] / (df['Parameters'] / 1000)  # loss per 1K params
            most_param_efficient = df.loc[df['Param Efficiency'].idxmin()]
            print(f"🎯 Most Parameter Efficient: {most_param_efficient['Experiment']} ({most_param_efficient['Param Efficiency']:.6f} loss/1K params)")

if __name__ == "__main__":
    main()
