# This program is used for transfer model imported from mxnet.model_zoo to numpy
# Note that there are alternative routes to translate the network

from mxnet.gluon import model_zoo, nn as m_nn
import numpy as np

import torchvision.models as models
from torch import nn as t_nn
import torch

pretrained_net = model_zoo.vision.vgg19(pretrained=True)
torch_vgg19 = models.vgg19(pretrained=True)

net = t_nn.Sequential()

for i in range(29):
  mxnet_layer = pretrained_net.features[i]
  torch_layer = torch_vgg19.features[i]
  if isinstance(mxnet_layer, m_nn.Conv2D):
    weight = mxnet_layer.weight.data().asnumpy()
    bias = mxnet_layer.bias.data().asnumpy()
    with torch.no_grad():
      torch_layer.weight.copy_(torch.tensor(weight))
      torch_layer.bias.copy_(torch.tensor(bias))
  net.add_module(f"layer {i}", torch_layer)
  print(f"finish upto layer: {i}")

print("finish transfering, writing to file param.log")
torch.save(net, 'param.log')