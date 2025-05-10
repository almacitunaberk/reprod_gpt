import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparams
batch_size = 64
block_size = 256 # also called context window for the transformer
max_iters = 5000
eval_interval = 300
lr = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200 # iterations per eval
n_embed = 384

torch.manual_seed(42)
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Since this is a character based autoregressive modeling, we'll get the 
#   unique characters to form the vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Mapping from characters to integers
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i,c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder behavior
decode = lambda idx: "".join([itos[i] for i in idx]) # decodes a given sequence of indices

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # 90% will be training, 10% will be validation
train_data = data[:n]
valid_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else valid_data
    ix = torch.randint(len(data)-block_size, (batch_size,)) # generating a tensor of shape (batch_size) with random indices used as starting indices
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

@torch.no_grad
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, n_embed, head_size, dropout=0.1):
        super().__init__()
        self.head_size = head_size
        self.w_k = nn.Linear(n_embed, head_size, bias=False)
        self.w_q = nn.Linear(n_embed, head_size, bias=False)
        self.w_v = nn.Linear(n_embed, head_size, bias=False)
        # Lower triangle matrix that we call tril here is not a model parameter as it's not trained
        #   thus, we need to register it as a buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.w_k(x)
        q = self.w_q(x)
        v = self.w_v(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, num_heads, head_size, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.heads = nn.ModuleList([Head(n_embed=n_embed, head_size=head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*head_size, n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        y = torch.cat([h(x) for h in self.heads], dim=-1)
        y = self.proj(y)
        y = self.dropout(y)
        return y

class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embed, num_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.self_attention = MultiHeadAttention(num_heads=num_heads, head_size=head_size, n_embed=n_embed)
        self.ffn = FeedForward(n_embed=n_embed)
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.ffn(self.layer_norm2(x))
        return x

# Bigram Model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embed, block_size, num_heads=6, num_blocks=6): # block_size is the sequence length
        super().__init__()
        # Since this is a character level transformer, given a character
        # the model needs to output probabilities for the next character
        # Embedding is just a lookup table -> given index j, it returns the row at that index j
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed=n_embed, num_heads=num_heads) for _ in range(num_blocks)])
        self.layer_norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
    
    def forward(self, x, targets=None):
        token_embeds = self.token_embedding_table(x)
        pos_range = torch.arange(self.block_size, device=device)
        pos_embeds = self.pos_emb(pos_range)
        embeds = token_embeds + pos_embeds
        x = self.blocks(embeds)
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.size()
            # normally, logits have shape (B,T,C), and targets have shape (B,T)
            # But, F.cross_entropy from torch requires (mini_batch_size, C, ...) shape
            # Thus, we need to properly change their shapes
            logits = logits.view(B*T, -1) 
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Given idx context (starting indices), we want to generate a total of max_new_tokens many
        #   characters produced by the model
        # Since this is inference, we need to generate the next character, append it to the idx
        #   and feed the model with the new context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # cropping only the last block_size many characters because the positional embedding will create a problem
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # only interested in the final character (the character produced by the model), shape is (B, C)
                                      # where C is channel, i.e. number of characters, i.e. vocab_size
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # shape is (B,1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
model = BigramLanguageModel(vocab_size=vocab_size, n_embed=n_embed, block_size=block_size, num_heads=8)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")
    
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, block_size), dtype=torch.long)
print(decode(model.generate(idx=context, max_new_tokens=1000)[0].tolist()))
    