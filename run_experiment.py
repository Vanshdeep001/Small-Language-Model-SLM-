#!/usr/bin/env python3
"""
Single Experiment Runner
Runs a single experiment with specified configuration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import tiktoken
import json
import time
from dataclasses import dataclass
from tqdm.auto import tqdm
from contextlib import nullcontext
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
import matplotlib.pyplot as plt

from experiment_config import get_experiment_config, ExperimentConfig

# Windows multiprocessing fix
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run a single SLM experiment')
    parser.add_argument('experiment_name', help='Name of the experiment to run')
    parser.add_argument('--skip-training', action='store_true', help='Skip training if model already exists')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        config = get_experiment_config(args.experiment_name)
    except ValueError as e:
        print(f"❌ Error: {e}")
        exit(1)
    
    print(f"🚀 Running experiment: {config.name}")
    print(f"📝 Description: {config.description}")
    print("=" * 60)
    
    # Install required packages
    print("Installing required packages...")
    os.system("pip install datasets tiktoken torch matplotlib")
    
    # Import after installation
    from datasets import load_dataset
    
    # Load dataset
    print("Loading TinyStories dataset...")
    try:
        ds = load_dataset("roneneldan/TinyStories")
        print(f"Dataset loaded successfully. Train samples: {len(ds['train'])}, Validation samples: {len(ds['validation'])}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check your internet connection and try again.")
        exit(1)
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    print(f"Tokenizer initialized. Vocab size: {enc.n_vocab}")
    
    # Model Architecture Components
    class LayerNorm(nn.Module):
        def __init__(self, ndim, bias):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(ndim))
            self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        
        def forward(self, x):
            return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

    class CausalSelfAttention(nn.Module):
        def __init__(self, config):
            super().__init__()
            assert config.n_embd % config.n_head == 0
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.attn_dropout = nn.Dropout(config.dropout)
            self.resid_dropout = nn.Dropout(config.dropout)
            self.n_head = config.n_head
            self.n_embd = config.n_embd
            self.flash = hasattr(F, 'scaled_dot_product_attention')
            if not self.flash:
                self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                           .view(1, 1, config.block_size, config.block_size))

        def forward(self, x):
            B, T, C = x.size()
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

            if self.flash:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True)
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self.resid_dropout(self.c_proj(y))
            return y

    class MLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.gelu = nn.GELU()
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            self.dropout = nn.Dropout(config.dropout)
        
        def forward(self, x):
            return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

    class Block(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.ln1 = LayerNorm(config.n_embd, config.bias)
            self.attn = CausalSelfAttention(config)
            self.ln2 = LayerNorm(config.n_embd, config.bias)
            self.mlp = MLP(config)

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    @dataclass
    class GPTConfig:
        block_size: int
        vocab_size: int
        n_layer: int
        n_head: int
        n_embd: int
        dropout: float = 0.1
        bias: bool = True

    class GPTModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            assert config.vocab_size is not None
            assert config.block_size is not None
            self.config = config

            self.transformer = nn.ModuleDict(dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, config.bias),
            ))
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.transformer.wte.weight = self.lm_head.weight

            # init all weights
            self.apply(self._init_weights)
            # apply special scaled init to the residual projections, per GPT-2 paper
            for pn, p in self.named_parameters():
                if pn.endswith('c_proj.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, idx, targets=None):
            device = idx.device
            b, t = idx.size()
            assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            pos = torch.arange(0, t, dtype=torch.long, device=device)

            tok_emb = self.transformer.wte(idx)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.ln_f(x)

            if targets is not None:
                logits = self.lm_head(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            else:
                logits = self.lm_head(x[:, [-1], :])
                loss = None

            return logits, loss

    # Check if model already exists
    model_path = f"best_model_{config.name}.pt"
    if os.path.exists(model_path) and args.skip_training:
        print(f"✅ Model {model_path} already exists. Skipping training.")
        exit(0)
    
    # Tokenize dataset if not already done
    train_data_path = f"train_{config.name}.bin"
    val_data_path = f"validation_{config.name}.bin"
    
    if not os.path.exists(train_data_path) or not os.path.exists(val_data_path):
        print("Tokenizing dataset...")
        
        def process_with_tokenizer(example):
            # Initialize tokenizer inside function for multiprocessing compatibility
            tokenizer = tiktoken.get_encoding("gpt2")
            ids = tokenizer.encode_ordinary(example['text'])
            out = {'ids': ids, 'len': len(ids)}
            return out
        
        # Use single process for Windows compatibility
        tokenized = ds.map(
            process_with_tokenizer,
            remove_columns=['text'],
            desc="tokenizing the splits",
            num_proc=1,  # Single process for Windows
        )
        
        # Apply dataset fraction
        if config.dataset_fraction < 1.0:
            train_size = int(len(tokenized['train']) * config.dataset_fraction)
            val_size = int(len(tokenized['validation']) * config.dataset_fraction)
            tokenized['train'] = tokenized['train'].select(range(train_size))
            tokenized['validation'] = tokenized['validation'].select(range(val_size))
            print(f"Using {config.dataset_fraction*100:.1f}% of dataset: {train_size} train, {val_size} validation samples")
        
        # Concatenate all the ids in each dataset into one large file
        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            filename = f'{split}_{config.name}.bin'
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            total_batches = 256  # Reduced for faster processing

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                try:
                    batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                    arr_batch = np.concatenate(batch['ids'])
                    arr[idx : idx + len(arr_batch)] = arr_batch
                    idx += len(arr_batch)
                except Exception as e:
                    if args.verbose:
                        print(f"Error processing batch {batch_idx}: {e}")
                    continue
            arr.flush()
            print(f"Saved {filename} with {idx} tokens")
    
    # Load data
    print("Loading training data...")
    train_data = np.memmap(train_data_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')
    
    # Create model
    model_config = GPTConfig(
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=0.1,
        bias=True
    )
    
    model = GPTModel(model_config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9)
    
    # Learning rate scheduler
    warmup_iters = min(100, config.max_iters // 10)
    lr_decay_iters = config.max_iters
    lr_decay_ratio = 0.1
    lr_decay_iters = config.max_iters
    min_lr = config.learning_rate * lr_decay_ratio
    
    def get_lr(it):
        if it < warmup_iters:
            return config.learning_rate * it / warmup_iters
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (config.learning_rate - min_lr)
    
    # Training loop
    print(f"Starting training for {config.max_iters} iterations...")
    start_time = time.time()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for iter_num in tqdm(range(config.max_iters), desc="Training"):
        # Learning rate scheduling
        lr = get_lr(iter_num)
        for param_group in optimizer.param_group:
            param_group['lr'] = lr
        
        # Training step
        model.train()
        batch = torch.randint(0, len(train_data) - config.block_size, (config.batch_size,), device=device)
        x = torch.stack([torch.from_numpy(train_data[i:i+config.block_size].astype(np.int64)) for i in batch])
        y = torch.stack([torch.from_numpy(train_data[i+1:i+config.block_size+1].astype(np.int64)) for i in batch])
        
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Validation
        if iter_num % 100 == 0 or iter_num == config.max_iters - 1:
            model.eval()
            with torch.no_grad():
                val_batch = torch.randint(0, len(val_data) - config.block_size, (config.batch_size,), device=device)
                val_x = torch.stack([torch.from_numpy(val_data[i:i+config.block_size].astype(np.int64)) for i in val_batch])
                val_y = torch.stack([torch.from_numpy(val_data[i+1:i+config.block_size+1].astype(np.int64)) for i in val_batch])
                val_logits, val_loss = model(val_x, val_y)
                val_losses.append(val_loss.item())
                
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    torch.save(model.state_dict(), model_path)
                
                if args.verbose or iter_num % 500 == 0:
                    print(f"Iter {iter_num}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}, lr={lr:.2e}")
    
    training_time = time.time() - start_time
    
    # Save results
    results = {
        "experiment_name": config.name,
        "description": config.description,
        "config": {
            "vocab_size": config.vocab_size,
            "n_embd": config.n_embd,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "block_size": config.block_size,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "max_iters": config.max_iters,
            "dataset_fraction": config.dataset_fraction
        },
        "results": {
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "training_time_seconds": training_time,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "best_val_loss": best_val_loss,
            "iterations": config.max_iters
        }
    }
    
    # Save results to JSON
    results_path = f"experiments/{config.name}_results.json"
    os.makedirs("experiments", exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create training plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', alpha=0.7)
    val_x = [i * 100 for i in range(len(val_losses))]
    plt.plot(val_x, val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(f'Training Progress - {config.name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = f"experiments/{config.name}_training_losses.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Experiment completed!")
    print(f"📊 Results saved to: {results_path}")
    print(f"📈 Plot saved to: {plot_path}")
    print(f"💾 Model saved to: {model_path}")
    print(f"⏱️  Training time: {training_time:.1f} seconds")
    print(f"📉 Best validation loss: {best_val_loss:.4f}")
    print(f"🔢 Total parameters: {results['results']['total_parameters']:,}")
