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
learning_rate = 1e-3
torch.manual_seed(1337)
block_size = 8
batch_size = 4
device = torch.device("cpu") if torch.backends.mps.is_available() else torch.device("cpu")
print(device)
eval_iters = 200
embed_dimen = 32
training_steps = 5000
num_heads = 4


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
        self.register_buffer('tril',torch.tril(torch.ones(embed_dimen,embed_dimen)))

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        query = self.query(x)
        wei = query @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # post matrix multiplication each element in the matrix is normalized by 1/sqrt(head_size). 
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei = F.softmax(wei,dim=1)
        value = self.value(x)
        out = wei @ value
        return out


class MultiHead(nn.Module):

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.sa_heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self,x):
        return torch.cat([h(x) for h in self.sa_heads],dim=-1)


class FeedForward(nn.Module):

    def __init__(self,embed_dimen):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dimen,embed_dimen),
            nn.ReLU()
        )

    def forward(self,x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self,num_heads,embed_dimen):
        super().__init__()
        self.sa_heads = MultiHead(num_heads,embed_dimen//num_heads)
        self.ffx = FeedForward(embed_dimen)

    def forward(self,x):
        x = self.sa_heads(x)
        out = self.ffx(x)
        return out

class BigramNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,embed_dimen)
        self.position_embedding_table = nn.Embedding(block_size,embed_dimen)
        self.sa_heads = MultiHead(num_heads, embed_dimen//num_heads)
        self.ffx = FeedForward(embed_dimen)
        self.block = Block(num_heads,embed_dimen)
        self.l_head = nn.Linear(embed_dimen,vocab_size)

    def forward(self,idx,targets=None):
        B,T = idx.shape
        token_embdding = self.token_embedding_table(idx) #(B,T,embed_dimen)
        position_embedding = self.position_embedding_table(torch.arange(T,device = device))#(B,T,embed_dimen)
        x = token_embdding + position_embedding #(B,T,embed_dimen)
        x = self.sa_heads(x)
        x = self.ffx(x)
        x = self.block(x)
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
    xt,yt = get_batch(True)
    logtis,loss = bm(xt,yt)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())


generated_tokens = bm.generate(prompt,max_tokens=100)
end = time.time()
print(f"Elapsed: {end - start:.4f}s")


print(decode(generated_tokens[0].tolist()))
