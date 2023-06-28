class Tokenizer:
    def __init__(self, samples) -> None:
        text = '0' + ''.join(samples)
        chars = sorted(list(set(text)))
        self.vocab_sz = len(chars)
        print(f'''alphabet:{''.join(chars)}''')
        print(f'''size:{self.vocab_sz}''')
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, x):    
        return [self.stoi[ch] for ch in x]
    
    def decode(self, x):
        return ''.join([self.itos[i] for i in x])
    