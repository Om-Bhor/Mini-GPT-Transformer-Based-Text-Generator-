import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cpu'  # safer for Mac

# Load checkpoint
checkpoint = torch.load("model.pt", map_location=device)
stoi = checkpoint['stoi']
itos = checkpoint['itos']
vocab_size = len(stoi)

# SAME hyperparameters as training
n_embd = 64
n_head = 4
n_layer = 2
block_size = 32
dropout = 0.2

# ---------------- MODEL ---------------- #

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B,T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln(x)
        return self.head(x)

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

# ---------------- LOAD MODEL ---------------- #

model = GPT().to(device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# ---------------- TOKENIZER ---------------- #

def encode(s):
    return [stoi[w] for w in s.split() if w in stoi]

def decode(l):
    return ' '.join([itos[i] for i in l])

# ---------------- STREAMLIT UI ---------------- #

st.title("Mini GPT 🤖")
st.write("Enter a prompt and generate text")

prompt = st.text_input("Prompt")

if st.button("Generate"):
    if prompt:
        tokens = encode(prompt)
        
        if len(tokens) == 0:
            st.write("⚠️ Words not in vocabulary. Try simpler words.")
        else:
            idx = torch.tensor([tokens], dtype=torch.long).to(device)
            out = model.generate(idx, max_new_tokens=50)
            result = decode(out[0].tolist())
            st.write(result)
    else:
        st.write("Please enter a prompt")