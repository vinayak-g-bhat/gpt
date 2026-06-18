from pathlib import Path
import urllib.request
import torch
from torch import nn
from torch.nn import functional as F
import time

input_file = Path("input.txt")

if not input_file.exists():
    try:
        urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', 'input.txt')
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")

input_text = ''
with open('input.txt','r',encoding='utf-8') as file:
    input_text = file.read()

vocab =  sorted(list(set(input_text)))
vocab_size = len(vocab)

ctoi = {ch:i for i,ch in enumerate(vocab) }
itoc = {i:ch for i,ch in enumerate(vocab)}
encode = lambda s:   [ctoi[ch] for ch in s]
decode = lambda l: ''.join([itoc[i] for i in l])

data = torch.tensor(encode(input_text),dtype=torch.long)
split_index = int(data.size(0) * 0.9)
train_data = data[:split_index] #90% of text is used for training
val_data = data[split_index:] #10% or left overs are kept asisde for validation

#hyperparameters
learning_rate = 1e-3
torch.manual_seed(1337)
block_size = 256
batch_size = 64
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(device)
eval_iters = 200
embed_dimen = 384
training_steps = 100
num_heads = 4
n_layers = 4
dropout = 0.2


def get_batch(isTraining):
    source = train_data if isTraining else val_data
    batch_offsets = torch.randint(0,len(source)-block_size,(batch_size,))
    x = torch.stack([source[batch_offset: batch_offset+block_size] for batch_offset in batch_offsets])
    y = torch.stack([source[1+batch_offset: batch_offset + block_size + 1] for batch_offset in batch_offsets]) 
    x = x.to(device)
    y = y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    bm.eval()
    for split in ["train","eval"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split == "train")
            logits, loss = bm(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    bm.train()
    return out


class Head(nn.Module):

    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(embed_dimen,head_size,bias=False)
        self.value = nn.Linear(embed_dimen,head_size,bias=False)
        self.query = nn.Linear(embed_dimen,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        query = self.query(x)
        wei = query @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # post matrix multiplication each element in the matrix is normalized by 1/sqrt(head_size). 
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei = F.softmax(wei,dim=1)
        wei = self.dropout(wei)
        value = self.value(x)
        out = wei @ value
        return out


class MultiHead(nn.Module):

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.sa_heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_dimen,embed_dimen)
        self.dropout= nn.Dropout(dropout)

    def forward(self,x):
        x= torch.cat([h(x) for h in self.sa_heads],dim=-1)
        x = self.dropout(self.proj(x))
        return x

class FeedForward(nn.Module):

    def __init__(self,embed_dimen):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dimen,4*embed_dimen),
            nn.ReLU(),
            nn.Linear(4*embed_dimen,embed_dimen),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self,num_heads,embed_dimen):
        super().__init__()
        self.sa_heads = MultiHead(num_heads,embed_dimen//num_heads)
        self.ffx = FeedForward(embed_dimen)
        self.norm1 = nn.LayerNorm(embed_dimen)
        self.norm2 = nn.LayerNorm(embed_dimen)

    def forward(self,x):
        x = x + self.sa_heads(self.norm1(x))
        out = x + self.ffx(self.norm2(x))
        return out

class BigramNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,embed_dimen)
        self.position_embedding_table = nn.Embedding(block_size,embed_dimen)
        self.blocks = nn.Sequential(*[Block(num_heads,embed_dimen) for _ in range(n_layers)])
        self.n_norm = nn.LayerNorm(embed_dimen)
        self.l_head = nn.Linear(embed_dimen,vocab_size)

    def forward(self,idx,targets=None):
        B,T = idx.shape
        token_embdding = self.token_embedding_table(idx) #(B,T,embed_dimen)
        position_embedding = self.position_embedding_table(torch.arange(T,device = device))#(B,T,embed_dimen)
        x = token_embdding + position_embedding #(B,T,embed_dimen)
        x = self.blocks(x)
        x = self.n_norm(x)
        logits = self.l_head(x) #(B,T,vocab_size)
        

        loss = None
        if targets != None:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets) 

        return logits,loss 
    
    def generate(self,prompt,max_tokens):
        for token in range(max_tokens):
            latest_prompt = prompt[:,-block_size:]
            logits,loss = self(latest_prompt) # __call__ method of nn.Module will be called
            logits = logits[:,-1,:]

            probalities = F.softmax(logits,dim=1)
            prediction = torch.multinomial(probalities,num_samples=1)
            prompt = torch.cat((prompt,prediction),dim=1)
        return prompt

start = time.time()


xt,yt = get_batch(True)

bm = BigramNeuralNetwork()
bm.to(device)
logits,loss = bm(xt,yt)

prompt = torch.zeros((1,1),dtype=torch.long,device=device)

optimizer = torch.optim.AdamW(bm.parameters(),lr=learning_rate)

batch_size = 32
for steps in range(training_steps):
    
    #calculate and log loss at regular intervals
    if steps % eval_iters == 0 or steps == training_steps - 1:
        losses = estimate_loss()
        print(f"step {steps}: training loss={losses['train']}, evaluation loss={losses['eval']}")

    xt,yt = get_batch(True)
    logtis,loss = bm(xt,yt)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print('basic loss' , loss.item())



generated_tokens = bm.generate(prompt,max_tokens=300)
end = time.time()
print(f"Elapsed: {end - start:.4f}s")


print(decode(generated_tokens[0].tolist()))
