import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import random

train_T = 6000
tot_T = 10000
epoch_num = 100
time = torch.arange(tot_T) / 1000
Y = time.sin() + torch.normal(0, 0.2, time.shape)

batch_size, num_steps = 32, 35

# sequential sampling
def sequential_sampling(dataset, batch_size, step_size):
  start = random.randint(0, step_size)
  num_tokens = ((len(dataset) - start - 1) // step_size // batch_size) * step_size * batch_size
  Xs = dataset[start : start + num_tokens].reshape((batch_size, -1))
  Ys = dataset[start + 1 : start + num_tokens + 1].reshape((batch_size, -1))
  for i in range(0, step_size * (Xs.shape[1] // step_size), step_size):
    X = Xs[:, i: i + step_size]
    Y = Ys[:, i: i + step_size]
    yield X, Y

def get_params(num_outputs, num_hiddens):
    num_inputs = num_outputs

    def normal(shape):
        return torch.randn(size=shape) * 0.01

    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens)
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens):
    return (torch.zeros((batch_size, num_hiddens)), )

def rnn(inputs, state, params):
    # Here `inputs` shape: (`num_steps`, `batch_size`, `vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for X in inputs:
        print(X.shape, W_xh.shape)
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

class RNNModelScratch: #@save
    """A RNN Model implemented from scratch."""
    def __init__(self, vocab_size, num_hiddens,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        return self.forward_fn(X.permute((1, 0)), state, self.params)

    def begin_state(self, batch_size):
        return self.init_state(batch_size, self.num_hiddens)

num_hiddens = 512
net = RNNModelScratch(1, num_hiddens, get_params,
                      init_rnn_state, rnn)

def grad_clipping(net, theta):  #@save
    """Clip the gradient."""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

#@save
def train_epoch_ch8(net, train_iter, loss, updater, use_random_iter):
    """Train a net within one epoch (defined in Chapter 8)."""
    state = None
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0])
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # `state` is a tensor for `nn.GRU`
                state.detach_()
            else:
                # `state` is a tuple of tensors for `nn.LSTM` and
                # for our custom scratch implementation
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # Since the `mean` function has been invoked
            updater(batch_size=1)
    return 

#@save
def train_ch8(net, train_iter, lr, num_epochs,
              use_random_iter=False):
    """Train a model (defined in Chapter 8)."""
    loss = nn.MSELoss()
    # Initialize
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    # Train and predict
    for epoch in range(num_epochs):
        train_epoch_ch8(
            net, train_iter, loss, updater, use_random_iter)


num_epochs, lr = 500, 1
train_ch8(net, sequential_sampling(Y[:train_T], batch_size, num_steps), lr, num_epochs)