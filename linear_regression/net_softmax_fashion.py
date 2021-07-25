import mxnet 
from mxnet import nd, autograd
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

# get training dataset
mnist_train = gdata.vision.FashionMNIST(train=True)
train_iter = gdata.DataLoader(mnist_train.transform_first(transformer), BATCH_SIZE, shuffle=True, last_batch='discard', num_workers=0)

# # print first 5 training data
# features, labels = mnist_train[:5]
# d2lzh.show_fashion_mnist(features, d2lzh.get_fashion_mnist_labels(labels))
# d2lzh.plt.show()

# get test dataset
mnist_test = gdata.vision.FashionMNIST(train=False)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer), BATCH_SIZE, shuffle=True, last_batch='discard', num_workers=0)

# define net structure
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize()
loss = gloss.SoftmaxCrossEntropyLoss()
trainer = mxnet.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': LEARNING_RATE})

# train
d2lzh.train_ch3(net, train_iter, test_iter, loss, ITERATION_COUNT, BATCH_SIZE, None, None, trainer)

