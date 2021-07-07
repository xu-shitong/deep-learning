from mxnet import nd, init
from mxnet.gluon import nn

'''
  1 will get.param initialize paramters? Yes, as it requires pass in shape parameters
  2 try get parameter fail
  3 try pre init parameters
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
net.add(Layer(2, 3), Layer(3, 2))
net.initialize()
print(net[1].weight)

# forward
print(net(nd.array(nd.arange(12).reshape((-1, 2)))))
