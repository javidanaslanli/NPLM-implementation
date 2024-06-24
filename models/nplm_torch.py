import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
from dataclasses import dataclass

@dataclass
class TestConfig:
    context_size: int = 3
    vocab_size: int = 7680
    embedding_size: int = 50
    hidden_size: int = 20
    learning_rate: float = 0.1
    num_epochs: int = 20000

with open('derslikler.txt', mode='r', encoding='utf-8') as f:
    text = f.read()
text = text[:150000] # For sake of efficiency

vocab_words = sorted(list(set(text.split())))
wordstoint = {wr:i for i,wr in enumerate(vocab_words)}
wordstoint['--'] = 5236
inttowords = {i:wr for wr,i in wordstoint.items()}
len(vocab_words)

def create_dataset(words, args: TestConfig):
    X, Y = [], []
    context = [5236] * args.context_size
    for word in words:
        if word in wordstoint:
            ix = wordstoint[word]
        else:
            ix = wordstoint["--"]  
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]
    
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    print(X.shape, Y.shape)
    return X, Y

words = text.split()
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

trwords = words[:n1]
valwords = words[n1:n2]
tewords = words[n2:]

Xtr, Ytr = create_dataset(trwords, TestConfig)
Xval, Yval = create_dataset(valwords, TestConfig)
Xte, Yte = create_dataset(tewords, TestConfig)

class Embeddings(nn.Module):
    def __init__(self, args: TestConfig):
        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings = args.vocab_size, embedding_dim=args.embedding_size)

    def forward(self, x):
        embedded = self.embeddings(x)
        flat_emb = embedded.reshape(embedded.shape[0], -1)
        return flat_emb
    
class MLP(nn.Module):
    def __init__(self, args: TestConfig):
        super().__init__()

        self.hidden_layer = nn.Linear(args.embedding_size * args.context_size, args.hidden_size, bias=False)
        self.output_layer = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def forward(self, x):

        x = self.hidden_layer(x)
        x = F.tanh(x)
        x = self.output_layer(x)

        return x

class NPLM(nn.Module):
    def __init__(self, args: TestConfig):
        super().__init__()

        self.emb = Embeddings(TestConfig)
        self.mlp = MLP(TestConfig)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
             nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.Embedding):
             nn.init.normal_(module.weight, mean=0, std = 0.02)

    def forward(self, x):

        x = self.emb(x)
        out = self.mlp(x)
        log_probs = F.log_softmax(out, dim = 1)

        return log_probs
    
model = NPLM(TestConfig)
optimizer = optim.SGD(params = model.parameters(), lr = TestConfig.learning_rate)
criterion = nn.NLLLoss()

for epoch in range(TestConfig.num_epochs):
    model.train()
    total_train_loss = 0
    optimizer.zero_grad()
    log_probs = model(Xtr)
    loss = criterion(log_probs, Ytr)
    loss.backward()
    optimizer.step()
    total_train_loss += loss.item()

    avg_train_loss = total_train_loss
    
    model.eval()
    total_val_loss = 0
 
    with torch.no_grad():
        log_probs = model(Xval)
        val_loss = criterion(log_probs, Yval)
        total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss
    
    if epoch % 1000 == 0: 
        print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")