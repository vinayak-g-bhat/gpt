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

ctoi = {ch:i for i,ch in enumerate(vocab) }
itoc = {i:ch for i,ch in enumerate(vocab)}
encode = lambda s:   [ctoi[ch] for ch in s]
decode = lambda l: ''.join([itoc[i] for i in l])

data = torch.tensor(encode(input_text),dtype=torch.long)
split_index = int(data.size(0) * 0.9)
train_data = data[:split_index] #90% of text is used for training
val_data = data[split_index:] #10% or left overs are kept asisde for validation

torch.manual_seed(1337)
block_size = 8
batch_size = 4
device = torch.device("cpu") if torch.backends.mps.is_available() else torch.device("cpu")
print(device)
eval_iters = 200

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


class BigramNeuralNetwork(nn.Module):

    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self,idx,targets=None):
        logits = self.token_embedding_table(idx) #(B,T,C)
        loss = None
        if targets != None:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets) 

        return logits,loss 
    
    def generate(self,prompt,max_tokens):
        for token in range(max_tokens):
            logits,loss = self(prompt) # __call__ method of nn.Module will be called
            logits = logits[:,-1,:]

            probalities = F.softmax(logits,dim=1)
            prediction = torch.multinomial(probalities,num_samples=1)
            prompt = torch.cat((prompt,prediction),dim=1)
        return prompt

start = time.time()


xt,yt = get_batch(True)

bm = BigramNeuralNetwork(len(vocab))
bm.to(device)
logits,loss = bm(xt,yt)

prompt = torch.zeros((1,1),dtype=torch.long,device=device)

optimizer = torch.optim.AdamW(bm.parameters())

batch_size = 32
for steps in range(100000):
    xt,yt = get_batch(True)
    logtis,loss = bm(xt,yt)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())


generated_tokens = bm.generate(prompt,max_tokens=300)
end = time.time()
print(f"Elapsed: {end - start:.4f}s")


print(decode(generated_tokens[0].tolist()))
