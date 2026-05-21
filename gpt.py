from pathlib import Path
import urllib.request
import torch

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
decode = lambda l: ''.join([itoc(i) for i in l])

data = torch.tensor(encode(input_text),dtype=torch.long)
split_index = int(data.size(0) * 0.9)
train_data = data[:split_index] #90% of text is used for training
val_data = data[split_index:] #10% or left overs are kept asisde for validation

torch.manual_seed(1337)
block_size = 8
batch_size = 4

def get_batch(isTraining):
    source = train_data if isTraining else val_data
    batch_offsets = torch.randint(0,len(source)-block_size,(batch_size,))
    x = torch.stack([source[batch_offset: batch_offset+block_size] for batch_offset in batch_offsets])
    y = torch.stack([source[1+batch_offset: batch_offset + block_size + 1] for batch_offset in batch_offsets]) 
    return x,y

xt,yt = get_batch(True)
print(xt)
print(yt)