#!/usr/bin/env python3
"""
SLM Story Generator
Generate stories using your trained Small Language Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import os
import math
import numpy as np
from dataclasses import dataclass

# Check if model exists
if not os.path.exists("best_model_params.pt"):
    print("❌ Model file 'best_model_params.pt' not found!")
    print("Please run training first.")
    exit(1)

print("✅ Loading your trained SLM for story generation...")

# Load tokenizer
enc = tiktoken.get_encoding("gpt2")

# Model configuration (matching the training script)
@dataclass
class ModelConfig:
    block_size: int = 64   # Actual trained model block size
    vocab_size: int = 50257
    n_layer: int = 4       # Actual trained model layers
    n_head: int = 4        # Actual trained model heads (256 % 4 = 0)
    n_embd: int = 256      # Actual trained model embedding dimension
    dropout: float = 0.1

config = ModelConfig()

# Model architecture (matching the training script)
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=True)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias=True)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, bias=True)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=True),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.shape
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

# Initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = GPTModel(config)
model.load_state_dict(torch.load("best_model_params.pt", map_location=device))
model.to(device)
model.eval()

print("✅ Model loaded successfully!")

def generate_story(prompt, max_length=150, temperature=0.8):
    """Generate a story from a prompt"""
    print(f"\n📝 Generating story from: '{prompt}'")
    print("=" * 60)
    
    # Encode prompt
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    generated_tokens = []
    with torch.no_grad():
        for i in range(max_length):
            # Get predictions
            logits, _ = model(tokens)
            logits = logits[:, -1, :] / temperature
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to sequence
            tokens = torch.cat([tokens, next_token], dim=1)
            generated_tokens.append(next_token.item())
            
            # Stop if we hit end token or max length
            if next_token.item() == enc.eot_token or len(generated_tokens) >= max_length:
                break
    
    # Decode generated text
    full_text = enc.decode(tokens[0].tolist())
    return full_text

# Interactive story generation
print("\n" + "="*60)
print("📚 SLM STORY GENERATOR")
print("="*60)
print("Your trained model is ready to generate stories!")
print()

# Pre-defined story prompts
story_prompts = [
    "Once upon a time there was a little girl",
    "A brave knight went on an adventure",
    "In a magical forest lived a friendly dragon",
    "The little boy found a mysterious box",
    "A cat sat on the mat and",
    "The princess was very sad because",
    "One day, a small mouse",
    "The magic wand could",
    "A little bird wanted to",
    "The old wizard had"
]

print("🎯 Generating sample stories...")
print()

for i, prompt in enumerate(story_prompts[:5], 1):
    story = generate_story(prompt, max_length=100)
    print(f"Story {i}:")
    print(story)
    print("\n" + "-" * 60)

print("\n🎉 Story generation complete!")
print("\n💡 Tips:")
print("• The model learned from TinyStories dataset (child-friendly content)")
print("• Try your own prompts like 'Once upon a time...' or 'A little girl...'")
print("• Adjust temperature (0.1-1.0) for more/less creative stories")
print("• Increase max_length for longer stories")

