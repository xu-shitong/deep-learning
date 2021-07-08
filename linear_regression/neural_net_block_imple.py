from mxnet import nd, init, autograd, gluon
from mxnet.gluon import nn, loss as gloss, data as gdata
import d2lzh

'''
  1 will get.param initialize paramters? Yes, as it requires pass in shape parameters
  2 try get parameter fail
  3 try pre init parameters
  4 try learning result
'''
class Layer(nn.Block):
  def __init__(self, node_in, node_out, **kwargs) -> None:
    super(Layer, self).__init__(**kwargs)
    self.weight = self.params.get('|weight_name|', shape=(node_in, node_out))

  def forward(self, x):
    # print(f"layer has paramter: {self.weight.data()}")
    return nd.dot(x, self.weight.data())

class indicator(init.Initializer):
  def __init__(self) -> None:
    super().__init__()
    self.count = 0

  def _init_weight(self, name, data):
    data[:] += self.count
    self.count += 1
    print('init_weight', name, data)

net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize()
# print(net[0].weight)

# # forward
# print(net(nd.array(nd.arange(12).reshape((-1, 2)))))

# training
# train_iter, test_iter = d2lzh.load_data_fashion_mnist(batch_size)

X = nd.arange(12).reshape((6, 2))
Y = nd.arange(6).reshape((6, ))
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
loss = gloss.SoftmaxCrossEntropyLoss()
train_iter = gdata.DataLoader(gdata.ArrayDataset(X, Y), 1, shuffle=False, num_workers = 0)

num_epochs = 5
d2lzh.train_ch3(net, train_iter, train_iter, loss, num_epochs, 1, None,
              None, trainer)


# for i in range(0, 7):
#   for X, y in train_iter:
#     with autograd.record():
#       y_hat = net(X)
#       loss = (y_hat - y) ** 2
#     loss.backward()

#     # params = net[0].weight
#     trainer.step(batch_size)
#     print(f"iteration{i+1} loss: {loss.mean().asscalar()}")
#     print(f"params: {net[0].weight.data()}")

