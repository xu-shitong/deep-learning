from mxnet import nd, init, gluon, autograd
from mxnet.gluon import nn, loss as gloss 
from mxnet.gluon import data as gdata

FEATURE_NUM = 2
SAMPLE_NUM = 1000
true_w = nd.array([2, -3.4])
true_b = 4.2
# generate training dataset
features = nd.random.normal(0, 1, shape=(SAMPLE_NUM, FEATURE_NUM))
labels = nd.dot(features, true_w.T)

labels = labels + true_b
labels += nd.random.normal(0, 0.01, shape=labels.shape)

ITERATION_NUM = 3
batch_size = 10

# setup neural network
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma = 0.3))
loss = gloss.L2Loss()
train = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
dataset = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

for i in range(1, ITERATION_NUM + 1):
  count = 0
  for X, y in data_iter:
    with autograd.record():
      l = loss(net(X), y)
    
    # print(f"loss vector = {l}")
    print(f"count={count}")
    l.backward()
    train.step(batch_size)
    count += 1

  l = loss(net(features), labels).mean().asnumpy()
  print('epoch %d, loss: %f' % (i, l))
print(net)