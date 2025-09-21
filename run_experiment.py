#!/usr/bin/env python3
"""
Experiment Runner for SLM
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
from experiment_config import ExperimentConfig, get_experiment_config, save_experiment_results

# Import required packages
try:
    from datasets import load_dataset
except ImportError:
    print("Installing required packages...")
    os.system("pip install datasets tiktoken torch matplotlib")
    from datasets import load_dataset

# Model Architecture Components (same as slm_final.py)
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
    dropout: float = 0.0
    bias: bool = True

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
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
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate tokens given a conditioning sequence."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # Ensure generated token is within vocabulary bounds
            idx_next = torch.clamp(idx_next, 0, self.config.vocab_size - 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def run_experiment(experiment_name: str, custom_config: ExperimentConfig = None):
    """Run a single experiment"""
    if custom_config:
        config = custom_config
    else:
        config = get_experiment_config(experiment_name)
    
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT: {config.name}")
    print(f"Description: {config.description}")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Vocabulary Size: {config.vocab_size:,}")
    print(f"  Embedding Dim: {config.n_embd}")
    print(f"  Layers: {config.n_layer}")
    print(f"  Attention Heads: {config.n_head}")
    print(f"  Block Size: {config.block_size}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Max Iterations: {config.max_iters}")
    print(f"  Dataset Subset: {config.dataset_subset*100:.1f}%")
    print(f"{'='*60}")

    # Load dataset
    print("Loading TinyStories dataset...")
    try:
        ds = load_dataset("roneneldan/TinyStories")
        print(f"Dataset loaded successfully. Train samples: {len(ds['train'])}, Validation samples: {len(ds['validation'])}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check your internet connection and try again.")
        return None

    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    print(f"Tokenizer initialized. Vocab size: {enc.n_vocab}")

    # Create custom tokenizer for reduced vocabulary if needed
    if config.vocab_size < enc.n_vocab:
        print(f"Creating custom tokenizer with vocab size {config.vocab_size}")
        # For simplicity, we'll use the first N tokens from GPT-2 vocab
        # In practice, you might want to create a proper custom tokenizer
        pass

    # Tokenize dataset if not already done
    train_file = f"train_{config.name}.bin"
    val_file = f"validation_{config.name}.bin"
    
    if not os.path.exists(train_file):
        print("Tokenizing dataset...")
        
        def process_with_tokenizer(example):
            ids = enc.encode_ordinary(example['text'])
            # Truncate vocabulary if needed
            if config.vocab_size < enc.n_vocab:
                ids = [min(id, config.vocab_size - 1) for id in ids]
            out = {'ids': ids, 'len': len(ids)}
            return out
        
        # Use subset of data if specified
        if config.dataset_subset < 1.0:
            train_size = int(len(ds['train']) * config.dataset_subset)
            val_size = int(len(ds['validation']) * config.dataset_subset)
            ds['train'] = ds['train'].select(range(train_size))
            ds['validation'] = ds['validation'].select(range(val_size))
            print(f"Using subset: {train_size} train, {val_size} validation samples")
        
        try:
            tokenized = ds.map(
                process_with_tokenizer,
                remove_columns=['text'],
                desc="tokenizing the splits",
                num_proc=1,  # Single process for Windows
            )
        except Exception as e:
            print(f"Error during tokenization: {e}")
            print("Trying with smaller batch size...")
            # Try with smaller dataset subset if tokenization fails
            if config.dataset_subset >= 0.1:
                config.dataset_subset = 0.1
                train_size = int(len(ds['train']) * config.dataset_subset)
                val_size = int(len(ds['validation']) * config.dataset_subset)
                ds['train'] = ds['train'].select(range(train_size))
                ds['validation'] = ds['validation'].select(range(val_size))
                print(f"Using smaller subset: {train_size} train, {val_size} validation samples")
                
                tokenized = ds.map(
                    process_with_tokenizer,
                    remove_columns=['text'],
                    desc="tokenizing the splits",
                    num_proc=1,
                )
            else:
                raise e
        
        # Concatenate all the ids in each dataset into one large file
        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            filename = f'{split}_{config.name}.bin'
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            
            # Use fewer batches for smaller datasets
            dataset_size = len(dset)
            total_batches = min(256, max(1, dataset_size // 100))  # Adaptive batch count

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                try:
                    batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                    arr_batch = np.concatenate(batch['ids'])
                    arr[idx : idx + len(arr_batch)] = arr_batch
                    idx += len(arr_batch)
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    continue
            arr.flush()
            print(f"Saved {filename} with {idx} tokens")

    # Device setup
    device = "cpu"  # Force CPU for Windows stability
    print(f"Using device: {device}")
    device_type = 'cpu'
    dtype = 'float32'
    ptdtype = torch.float32
    ctx = nullcontext()

    torch.set_default_device(device)
    torch.manual_seed(42)

    # Model configuration
    model_config = GPTConfig(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        bias=config.bias
    )

    # Create model
    model = GPT(model_config)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")

    # Training parameters
    warmup_steps = min(200, config.max_iters // 10)
    min_lr = config.learning_rate / 10
    eval_iters = max(1, min(50, config.max_iters // 40))  # Ensure eval_iters is at least 1
    gradient_accumulation_steps = max(1, 32 // config.batch_size)

    # Batch generation function
    def get_batch(split):
        """Generate a batch of data"""
        if split == 'train':
            data = np.memmap(train_file, dtype=np.uint16, mode='r')
        else:
            data = np.memmap(val_file, dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    # Loss estimation function
    def estimate_loss(model):
        """Estimate loss on train and validation sets"""
        out = {}
        model.eval()
        with torch.inference_mode():
            for split in ['train', 'val']:
                losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    X, Y = get_batch(split)
                    with ctx:
                        logits, loss = model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
        model.train()
        return out

    # Optimizer and scheduler setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9)
    scheduler_warmup = LinearLR(optimizer, total_iters=warmup_steps)
    scheduler_decay = CosineAnnealingLR(optimizer, T_max=config.max_iters - warmup_steps, eta_min=min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])

    # Training loop
    print("Starting training...")
    start_time = time.time()
    best_val_loss = float('inf')
    best_model_params_path = f"best_model_{config.name}.pt"
    train_loss_list, validation_loss_list = [], []

    for epoch in tqdm(range(config.max_iters), desc="Training"):
        if epoch % eval_iters == 0 and epoch != 0:
            losses = estimate_loss(model)
            print(f"\nEpoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.5f}")
            train_loss_list.append(losses['train'])
            validation_loss_list.append(losses['val'])

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), best_model_params_path)
                print(f"New best model saved! Val loss: {best_val_loss:.4f}")

        # Training step
        X, y = get_batch("train")
        X, y = X.to(device), y.to(device)

        with ctx:
            logits, loss = model(X, y)
            loss = loss / gradient_accumulation_steps
            loss.backward()

        if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == config.max_iters):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds!")

    # Test inference
    print("Testing model inference...")
    model.load_state_dict(torch.load(best_model_params_path, map_location=torch.device(device)))
    model.eval()

    # Test prompts
    test_prompts = [
        "Once upon a time there was a pumpkin.",
        "A little girl went to the woods",
        "The cat sat on the mat and",
        "In a small village lived a"
    ]

    print(f"\n{'='*60}")
    print(f"EXPERIMENT RESULTS: {config.name}")
    print(f"{'='*60}")

    generated_samples = []
    for i, sentence in enumerate(test_prompts, 1):
        print(f"\nPrompt {i}: {sentence}")
        try:
            # Encode prompt and ensure tokens are within vocabulary bounds
            tokens = enc.encode_ordinary(sentence)
            if config.vocab_size < enc.n_vocab:
                tokens = [min(token, config.vocab_size - 1) for token in tokens]
            
            context = torch.tensor(tokens).unsqueeze(dim=0).to(device)
            
            with torch.no_grad():
                y = model.generate(context, 50, temperature=0.8, top_k=50)
            
            # Ensure generated tokens are within bounds before decoding
            generated_tokens = y.squeeze().tolist()
            if config.vocab_size < enc.n_vocab:
                generated_tokens = [min(token, config.vocab_size - 1) for token in generated_tokens]
            
            generated_text = enc.decode(generated_tokens)
            print(f"Generated: {generated_text}")
            generated_samples.append(generated_text)
        except Exception as e:
            print(f"Error generating text: {e}")
            generated_samples.append(f"Error: {str(e)}")
        print("-" * 60)

    # Compile results
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
            "dropout": config.dropout,
            "dataset_subset": config.dataset_subset
        },
        "model_info": {
            "total_parameters": total_params,
            "training_time_seconds": training_time
        },
        "training_results": {
            "final_train_loss": float(train_loss_list[-1]) if train_loss_list else None,
            "final_val_loss": float(validation_loss_list[-1]) if validation_loss_list else None,
            "best_val_loss": float(best_val_loss),
            "train_losses": [float(x) for x in train_loss_list],
            "val_losses": [float(x) for x in validation_loss_list]
        },
        "generated_samples": generated_samples
    }

    # Save results
    save_experiment_results(config.name, results)
    
    # Plot training losses
    if train_loss_list and validation_loss_list:
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_list, 'g', label='train_loss')
        plt.plot(validation_loss_list, 'r', label='validation_loss')
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss - {config.name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'experiments/{config.name}_training_losses.png')
        plt.close()

    print(f"\n{'='*60}")
    print(f"EXPERIMENT {config.name} COMPLETED!")
    print(f"Best model saved as: {best_model_params_path}")
    print(f"Results saved to experiments/{config.name}_results.json")
    print(f"Training plot saved as experiments/{config.name}_training_losses.png")
    print(f"{'='*60}")

    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python run_experiment.py <experiment_name>")
        print("\nAvailable experiments:")
        from experiment_config import EXPERIMENT_CONFIGS
        for name, config in EXPERIMENT_CONFIGS.items():
            print(f"  {name:20} - {config.description}")
        sys.exit(1)
    
    experiment_name = sys.argv[1]
    try:
        results = run_experiment(experiment_name)
        if results:
            print(f"\nExperiment '{experiment_name}' completed successfully!")
        else:
            print(f"\nExperiment '{experiment_name}' failed!")
    except Exception as e:
        print(f"Error running experiment '{experiment_name}': {e}")
        import traceback
        traceback.print_exc()