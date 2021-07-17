from numpy.core import numeric
import d2lzh 
from mxnet import nd, init, image, gluon
from mxnet.gluon import model_zoo, nn, loss as gloss, data as gdata
import numpy as np

def bilinear_kernel(in_channels, out_channels, kernel_size): 
  factor = (kernel_size + 1) // 2
  if kernel_size % 2 == 1: 
    center = factor - 1
  else:
    center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
          (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
  return nd.array(weight)

## FCN network part
net = nn.HybridSequential()

# 1. get pretrained ResNet network
pretrained_net = model_zoo.vision.resnet18_v2(pretrained=True)
print(pretrained_net.features[-4:])

# create net and add layers from ResNet
for layer in pretrained_net.features[:-2]:
  net.add(layer)

# 2. create (1, 1) convolusion network part
channel_num = 21
net.add(nn.Conv2D(channel_num, kernel_size=(1, 1)))
net[-1].initialize(init=init.Xavier())

# 3. add transpose convolusion layer
net.add(nn.Conv2DTranspose(channel_num, kernel_size=64, padding=16, strides=32))
net[-1].initialize(init=init.Constant(bilinear_kernel(channel_num, channel_num, 64)))


voc_dir = "../voc-2012/VOCdevkit/VOC2012"
crop_size, batch_size, colormap2label = (320, 480), 32, nd.zeros(256**3)
train_iter = gdata.DataLoader(
             d2lzh.VOCSegDataset(True, crop_size, voc_dir, colormap2label), batch_size,
             shuffle=True, last_batch='discard')
test_iter = gdata.DataLoader(
             d2lzh.VOCSegDataset(False, crop_size, voc_dir, colormap2label), batch_size,
             last_batch='discard')
loss = gloss.SoftmaxCrossEntropyLoss(axis=1)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1,
                                                      'wd': 1e-3})
d2lzh.train(train_iter, test_iter, net, loss, trainer, ctx=d2lzh.try_gpu(), num_epochs=5)



# get image, trial on transpose convolusion network
img = image.imread('../voc-2012/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg')
X = img.astype('float32').transpose((2, 0, 1)).expand_dims(axis=0) / 255
Y = net(X)
out_img_index = nd.argmax(Y[0], axis=0).astype('int32')
out_img = nd.array(d2lzh.VOC_COLORMAP)[out_img_index].astype('uint8')

d2lzh.set_figsize()
print('input image shape:', img.shape) 
d2lzh.plt.imshow(img.asnumpy())

print('output image shape:', out_img.shape) 
d2lzh.plt.imshow(out_img.asnumpy())
d2lzh.plt.show()