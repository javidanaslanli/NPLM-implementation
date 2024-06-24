import jaxlib
import jax.numpy as jnp
from collections import Counter
from dataclasses import dataclass
import flax.linen as nn
from flax.linen.initializers import normal
from jax import random
import numpy as np
import jax.nn

@dataclass
class TestConfig:
    context_size: int = 5
    vocab_size: int = 7680
    embedding_size: int = 30
    hidden_size: int = 80
    learning_rate: float = 0.1
    num_epochs: int = 12000

with open('derslikler.txt', mode='r', encoding='utf-8') as f:
    text = f.read()

text = text[:150000]


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
    
    X = jnp.array(X, dtype=jnp.uint16)
    Y = jnp.array(Y, dtype=jnp.uint16)
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

class NPLM(nn.Module):

    vocab_size: int
    emb_size: int
    hidden_size: int
    context_size: int

    def setup(self):
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.emb_size,
                                    embedding_init=normal(stddev=0.01))
        self.hidden_layer = nn.Dense(self.hidden_size, kernel_init=normal(stddev=0.01))
        self.output_layer = nn.Dense(self.vocab_size, kernel_init=normal(stddev=0.01))

    def __call__(self, x):
        embedded = self.embedding(x)
        flattened = embedded.reshape(embedded.shape[0], -1)
        hidden = jnp.tanh(self.hidden_layer(flattened))
        logits = self.output_layer(hidden)

        return logits
    
model = NPLM(
    vocab_size=TestConfig.vocab_size,
    emb_size=TestConfig.embedding_size,
    hidden_size = TestConfig.hidden_size,
    context_size= TestConfig.context_size
)

key = random.PRNGKey(0)
params = model.init(key, jnp.ones((1, TestConfig.context_size), dtype=jnp.int16))
output = model.apply(params, Xtr)

def cross_entropy_loss(logits, targets):
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(log_probs[jnp.arange(logits.shape[0]), targets])

@jax.jit
def update_step(params, inputs, targets, learning_rate=TestConfig.learning_rate):
    def loss_fn(params):
        logits = model.apply(params, inputs)
        return cross_entropy_loss(logits, targets)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(params)
    new_params = jax.tree.map(lambda p, g: p - learning_rate * g, params, grads)

    return new_params, loss

def train(params, num_epochs, X, Y):
    batch_size = X.shape[0]
    for epoch in range(num_epochs):
        params, loss = update_step(params, X, Y)
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    return params   

trained_params = train(params, num_epochs=TestConfig.num_epochs, X=Xtr, Y=Ytr)