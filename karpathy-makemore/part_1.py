import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 

WORDS = open('names.txt','r').read().splitlines()
_chars = sorted(list(set(''.join(WORDS))))
STOI = {s:i+1 for i,s in enumerate(_chars)}
STOI['.'] = 0
ITOS = { i:s for s,i in STOI.items()}

def main():
    N = torch.zeros((len(STOI),len(STOI)),dtype=torch.int32)

    # Count bigrams in 2D array N[left, right]
    for w in WORDS:
        chs = ['.'] + list(w) + ['.']
        for ch1,ch2 in zip(chs,chs[1:]):
            ix1 = STOI[ch1]
            ix2 = STOI[ch2]
            N[ix1,ix2] += 1
    
    #visualize(N)

    # Normalize the counts to get probabilities
    dist = (N+1).float() # plus one smoothing
    dist /= dist.sum(1, keepdim=True)

    #generate(dist,5)

    loss(WORDS, dist)

def bigram_dist(words):
    # Count bigrams in 2D array N[left, right]
    N = torch.zeros((len(STOI),len(STOI)),dtype=torch.int32)
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1,ch2 in zip(chs,chs[1:]):
            ix1 = STOI[ch1]
            ix2 = STOI[ch2]
            N[ix1,ix2] += 1

    dist = (N+1).float() # plus one smoothing
    dist /= dist.sum(1, keepdim=True)
    return dist

def loss(words, dist):
    log_likelyhood = 0
    n = 0
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1,ch2 in zip(chs,chs[1:]):
            ix1 = STOI[ch1]
            ix2 = STOI[ch2]
            prob = dist[ix1,ix2]
            logprob = torch.log(prob)
            log_likelyhood += logprob
            n += 1
            #print(f'{ch1}->{ch2}: {prob:.4f}')
    nll = -log_likelyhood / n        
    print(f'{nll=}')
    return nll

def generate(dist, nsamps):
    g = torch.Generator().manual_seed(2147483647)
    prev = 0
    for _ in range(nsamps):
        seq = []
        while 1:
            seq.append(prev)
            prev = torch.multinomial(dist[prev], num_samples=1, replacement=True, generator=g).item()
            if prev == 0: break
        print(''.join([ITOS[i] for i in seq]))

def visualize(N):
    # Visualize the bigram counts
    plt.figure(figsize=(14,14))
    plt.imshow(N, cmap='Blues')
    for i in range(len(STOI)):
        for j in range(len(STOI)):
            chstr = ITOS[i] + ITOS[j]
            plt.text(j,i,chstr,ha='center',va='bottom',color='gray')
            plt.text(j,i,N[i,j].item(),ha='center',va='top',color='gray')
    plt.axis('off')
    plt.show()

def build_bigram_training_set(words):
    xs = []
    ys = []
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1,ch2 in zip(chs,chs[1:]):
            ix1 = STOI[ch1]
            ix2 = STOI[ch2]
            xs.append(ix1)
            ys.append(ix2)
    xenc = F.one_hot(torch.tensor(xs), num_classes=len(STOI)).float()
    yenc = F.one_hot(torch.tensor(ys), num_classes=len(STOI)).float()        
    return xs, ys, xenc, yenc

if __name__ == '__main__':
    xs, ys, xenc, yenc = build_bigram_training_set(WORDS)
    #plt.imshow(xenc)
    #plt.show()
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((len(STOI),len(STOI)), generator=g, requires_grad=True)

    for _ in range(100):
        # forward pass
        logits = xenc @ W # log-counts
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)
        loss = -probs[torch.arange(len(xs)), ys].log().mean()
        print(f'{loss=}')
        W.grad = None
        # backward pass
        loss.backward()
        # update
        W.data -= 50 * W.grad

    p = probs
    print(probs[0])
    b = bigram_dist(WORDS)
    print(b[0])

    #main()