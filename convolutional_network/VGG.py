import mxnet 
from mxnet import nd, autograd, init
from mxnet.gluon import data as gdata, loss as gloss, nn
import d2lzh
import sys

# define super parameters
BATCH_SIZE = 10
INPUT_NUM = 28 * 28
OUTPUT_NUM = 10
LEARNING_RATE = 0.03
ITERATION_COUNT = 5
if sys.platform.startswith('win'):
  num_workers = 0 # 0表示不用额外的进程来加速读取数据 
else:
  num_workers = 4

transformer = gdata.vision.transforms.ToTensor()

# get dataset
train_iter, test_iter = d2lzh.load_data_fashion_mnist(BATCH_SIZE, resize=224)

# define network structure
net = nn.Sequential()

def add_vgg(conv_num, channel_num):
  for i in range(conv_num):
    net.add(nn.Conv2D(channels=channel_num, kernel_size=3, padding=1))
  net.add(nn.MaxPool2D(pool_size=2, strides=2))

vgg_structure = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]

for conv_num, channel_num in vgg_structure:
  add_vgg(conv_num, channel_num)

net.add(nn.Dense(4096, activation="relu"), nn.Dropout(0.5),
        nn.Dense(4096, activation="relu"), nn.Dropout(0.5),
        nn.Dense(10))
net.initialize(init=init.Xavier())

# train
loss = gloss.SoftmaxCrossEntropyLoss()
trainer = mxnet.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': LEARNING_RATE})

d2lzh.train_ch3(net, train_iter, test_iter, loss, ITERATION_COUNT, BATCH_SIZE, None, None, trainer)
