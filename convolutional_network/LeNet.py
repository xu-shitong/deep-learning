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

# define network structure
net = nn.Sequential()
net.add(
        # network layers
        nn.Conv2D(6, kernel_size=5, activation='sigmoid'),
        nn.MaxPool2D((2,2), strides=2),
        nn.Conv2D(16, kernel_size=5, activation='sigmoid'),
        nn.MaxPool2D((2,2), strides=2),
        # dense layers
        nn.Dense(120, activation='sigmoid'),
        nn.Dense(84, activation='sigmoid'),
        nn.Dense(10))

net.initialize()

# train
loss = gloss.SoftmaxCrossEntropyLoss()
trainer = mxnet.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': LEARNING_RATE})

for i in range(1):
  for X, y in train_iter:
    with autograd.record():
      l = loss(net(X), y)

    l.backward()
    trainer.step(BATCH_SIZE)
    break

d2lzh.train_ch3(net, train_iter, test_iter, loss, ITERATION_COUNT, BATCH_SIZE, None, None, trainer)
