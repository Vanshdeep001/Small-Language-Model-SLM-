#!/usr/bin/env python3
"""
Simple SLM Testing Script
Test your trained Small Language Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import os
import math
from dataclasses import dataclass

# Check if model exists
if not os.path.exists("best_model_params.pt"):
    print("❌ Model file 'best_model_params.pt' not found!")
    print("Please run training first.")
    exit(1)

print("✅ Model file found!")
print("Loading your trained SLM...")

# Load tokenizer
enc = tiktoken.get_encoding("gpt2")

# Model configuration (same as training)
@dataclass
class ModelConfig:
    block_size: int = 128
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1

config = ModelConfig()

# Simple GPT model (same as training)
class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

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

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            act     = nn.GELU(),
            dropout = nn.Dropout(config.dropout),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Causal mask
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

# Initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = GPTModel(config)
model.load_state_dict(torch.load("best_model_params.pt", map_location=device))
model.to(device)
model.eval()

print("✅ Model loaded successfully!")

def generate_text(prompt, max_length=100, temperature=0.8):
    """Generate text from a prompt"""
    # Encode prompt
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    generated = []
    with torch.no_grad():
        for _ in range(max_length):
            # Get predictions
            logits, _ = model(tokens)
            logits = logits[:, -1, :] / temperature
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to sequence
            tokens = torch.cat([tokens, next_token], dim=1)
            generated.append(next_token.item())
            
            # Stop if we hit end token or max length
            if next_token.item() == enc.eot_token or len(generated) >= max_length:
                break
    
    # Decode generated text
    full_text = enc.decode(tokens[0].tolist())
    return full_text

# Interactive testing
print("\n" + "="*60)
print("🎯 SLM TESTING INTERFACE")
print("="*60)
print("Your model is ready! Try these test prompts:")
print()

test_prompts = [
    "Once upon a time",
    "The little girl",
    "A cat sat",
    "In the forest",
    "The magic"
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"Test {i}: '{prompt}'")
    generated = generate_text(prompt, max_length=50)
    print(f"Generated: {generated}")
    print("-" * 40)

print("\n🎉 Your SLM is working perfectly!")
print("You can modify the test_prompts list to try your own prompts.")
