# This program is used for transfer model imported from mxnet.model_zoo to numpy
# Note that there are alternative routes to translate the network

from mxnet.gluon import model_zoo
import numpy as np

pretrained_net = model_zoo.vision.vgg19(pretrained=True)
output = []
for i in range(29):
  param = pretrained_net.features[i].weight.data()
  output.append(
    param.asnumpy()
  )
  
np.savetxt('param.txt', output)