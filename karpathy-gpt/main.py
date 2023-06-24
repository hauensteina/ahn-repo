
import torch 
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from bigram import BigramLanguageModel
from transformer import TransformerModel
from transformer import BLOCK_SZ, EMBED_SZ, BATCH_SZ
from tokenizer import Tokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 3e-4
EVAL_INTERVAL = 500
N_EPOCHS = 5000

def read_data():
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    print('length of text: ', len(text))
    print(text[:1000])
    return text

def split_train_val(data):
    train_frac = 0.9
    n = int(train_frac * len(data))
    train_data = data[:n]
    val_data = data[n:] 
    return train_data, val_data

def get_batch(data,split):
    data = split_train_val(data)[0] if split == 'train' else split_train_val(data)[1]
    ix = torch.randint(0, len(data) - BLOCK_SZ, (BATCH_SZ,))
    x = torch.stack([data[i:i+BLOCK_SZ] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SZ+1] for i in ix])
    x,y = x.to(DEVICE), y.to(DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss(m,data):
    """ Runs a few batches through the model and returns the average train and val loss"""
    eval_batches = 100
    out = {}
    m.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_batches)
        for k in range(eval_batches):
            x,y = get_batch(data,split)
            logits, loss = m(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

def main():
    text = read_data()
    tok = Tokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    print(data.shape, data.dtype)
    print(data[:100])   
    train_data, val_data = split_train_val(data)

    torch.manual_seed(1337)
    xb, yb = get_batch(data,'train')

    #model = BigramLanguageModel(tok, DEVICE)
    model = TransformerModel(tok, DEVICE)
    m = model.to(DEVICE)

    logits, loss = m(xb, yb)
    print(logits.shape)
    print(loss)

    print(m.generate_one(100))

    optimizer = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE) # 3e-4 for larger nets
    
    for iter in range(N_EPOCHS):
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(m,data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb,yb = get_batch(data,'train')
        logits,loss = m(xb,yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    print(loss.item())
    print(m.generate_one(500))




    tt=42


main()