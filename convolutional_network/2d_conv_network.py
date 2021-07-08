from mxnet import nd, autograd
from mxnet.gluon import nn



X = nd.ones((6, 8))
X[:, 2:6] = 0

Y = nd.array(
  [ [0, 1, 0, 0, 0, -1, 0],
    [0, 1, 0, 0, 0, -1, 0] ,
    [0, 1, 0, 0, 0, -1, 0] ,
    [0, 1, 0, 0, 0, -1, 0] ,
    [0, 1, 0, 0, 0, -1, 0] ,
    [0, 1, 0, 0, 0, -1, 0] ])

X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

lr = 0.03

cov = nn.Conv2D(1, kernel_size=(1,2))
cov.initialize()

B = nd.zeros((6, 7))
B.attach_grad()

for i in range(10):
  with autograd.record():
    y_hat = cov(X) + B
    loss = (Y - y_hat) ** 2
  loss.backward()

  cov.weight.data()[:] -= lr * cov.weight.grad()
  B -= lr * B.grad
  print(f"iteration {i+1} loss: {loss.norm().asscalar()}")
print(cov.weight.data())
print(B)