#!/usr/bin/env python3
"""
Quick Start Script for SLM Experiments
Shows you how to get started with experiments
"""

import os
import sys

def print_welcome():
    """Print welcome message and instructions"""
    print("\n" + "="*80)
    print("🚀 WELCOME TO SLM EXPERIMENT FRAMEWORK")
    print("="*80)
    print("This framework lets you experiment with different model configurations:")
    print("• Vocabulary sizes (10K, 25K, 50K tokens)")
    print("• Embedding dimensions (128, 256, 512, 768)")
    print("• Model architectures (layers, attention heads)")
    print("• Training data sizes")
    print("• Learning rates")
    print("="*80)

def print_quick_start():
    """Print quick start instructions"""
    print("\n📋 QUICK START GUIDE")
    print("-" * 40)
    print("1. Run a single quick experiment:")
    print("   python demo_experiments.py single")
    print()
    print("2. Run 3 key experiments (30-60 minutes):")
    print("   python demo_experiments.py demo")
    print()
    print("3. Interactive experiment runner:")
    print("   python experiment_runner.py")
    print()
    print("4. Run vocabulary experiments:")
    print("   python run_batch_experiments.py vocab")
    print()
    print("5. Compare all results:")
    print("   python run_batch_experiments.py compare")

def print_experiment_examples():
    """Print example experiments"""
    print("\n🔬 EXAMPLE EXPERIMENTS")
    print("-" * 40)
    print("Vocabulary Size Experiments:")
    print("  • vocab_small  - 10K tokens (faster training)")
    print("  • vocab_medium - 25K tokens (balanced)")
    print("  • baseline     - 50K tokens (full vocabulary)")
    print()
    print("Embedding Dimension Experiments:")
    print("  • embd_small   - 128 dims (smaller model)")
    print("  • baseline     - 256 dims (current best)")
    print("  • embd_large   - 512 dims (more capacity)")
    print()
    print("Quick Experiments:")
    print("  • optimal_fast - Fast training (1000 iterations)")
    print("  • data_small   - 10% of training data")

def print_what_to_expect():
    """Print what to expect from experiments"""
    print("\n📊 WHAT TO EXPECT")
    print("-" * 40)
    print("Each experiment will:")
    print("• Train a model with specific configuration")
    print("• Generate sample text outputs")
    print("• Save model checkpoint and results")
    print("• Create training loss plots")
    print()
    print("Batch experiments will:")
    print("• Run multiple experiments automatically")
    print("• Compare results in tables and plots")
    print("• Identify best configurations")
    print()
    print("Typical results:")
    print("• Training time: 5-60 minutes per experiment")
    print("• Model size: 1M-10M parameters")
    print("• Validation loss: 2.0-4.0 (lower is better)")

def print_troubleshooting():
    """Print troubleshooting tips"""
    print("\n🔧 TROUBLESHOOTING")
    print("-" * 40)
    print("If experiments fail:")
    print("• Check internet connection (needs to download dataset)")
    print("• Ensure you have 4-8GB RAM available")
    print("• Try smaller experiments first (optimal_fast)")
    print("• Check disk space (need ~2GB)")
    print()
    print("If training is slow:")
    print("• Use smaller models (embd_small, vocab_small)")
    print("• Reduce max_iters in experiment config")
    print("• Use data_small for faster training")

def main():
    """Main function"""
    print_welcome()
    
    if len(sys.argv) > 1 and sys.argv[1] == "help":
        print_quick_start()
        print_experiment_examples()
        print_what_to_expect()
        print_troubleshooting()
        return
    
    print_quick_start()
    
    print("\n❓ Need more help?")
    print("Run: python start_experiments.py help")
    print("Or check: EXPERIMENT_README.md")
    
    print("\n🎯 RECOMMENDED FIRST STEPS:")
    print("1. python demo_experiments.py single    # Quick test (5-15 min)")
    print("2. python demo_experiments.py demo      # Key experiments (30-60 min)")
    print("3. python run_batch_experiments.py compare  # Compare results")
    
    print("\n" + "="*80)
    print("Happy experimenting! 🚀")
    print("="*80)

if __name__ == "__main__":
    main()

