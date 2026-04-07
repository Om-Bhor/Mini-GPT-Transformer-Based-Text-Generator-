import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset

# Hyperparameters
batch_size = 8
block_size = 32
max_iters = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 64
n_head = 4
n_layer = 2
dropout = 0.2

torch.manual_seed(1337)

# Load dataset (WikiText)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
text = " ".join(dataset['train']['text'])

# Tokenization (simple word-level)
words = text.split()
vocab = sorted(set(words))
vocab_size = len(vocab)

stoi = {w:i for i,w in enumerate(vocab)}
itos = {i:w for i,w in enumerate(vocab)}

encode = lambda s: [stoi[w] for w in s.split() if w in stoi]
decode = lambda l: ' '.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# Train/Val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Model components
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

    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.head(x)

        if targets is None:
            return logits, None

        B,T,C = logits.shape
        loss = F.cross_entropy(logits.view(B*T,C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits,_ = self(idx_cond)
            probs = F.softmax(logits[:,-1,:], dim=-1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

model = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for i in range(max_iters):
    xb,yb = get_batch('train')
    logits,loss = model(xb,yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print(f"Step {i}, Loss: {loss.item()}")

# Save model + vocab
torch.save({
    'model_state': model.state_dict(),
    'stoi': stoi,
    'itos': itos
}, "model.pt")

print("Training complete & model saved!")