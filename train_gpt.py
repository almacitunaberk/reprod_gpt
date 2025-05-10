import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparams
batch_size = 32
block_size = 8 # also called context window for the transformer
max_iters = 30000
eval_interval = 300
lr = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200 # iterations per eval

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

# Bigram Model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Since this is a character level transformer, given a character
        # the model needs to output probabilities for the next character
        # Embedding is just a lookup table -> given index j, it returns the row at that index j
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) 
    
    def forward(self, x, targets=None):
        logits = self.token_embedding_table(x)
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
            logits, loss = self(idx)
            logits = logits[:, -1, :] # only interested in the final character (the character produced by the model), shape is (B, C)
                                      # where C is channel, i.e. number of characters, i.e. vocab_size
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # shape is (B,1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
model = BigramLanguageModel(vocab_size)
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

context = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))
    