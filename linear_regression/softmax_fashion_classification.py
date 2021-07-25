import mxnet 
from mxnet import nd, autograd
from mxnet.gluon import data as gdata
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
W = nd.random.normal(0, 1, shape=(INPUT_NUM, OUTPUT_NUM))
W.attach_grad()
b = nd.zeros(OUTPUT_NUM)
b.attach_grad()

# define softmax function
def softmax(X):
  exp_x = X.exp()
  # exp_sum = exp_x.sum()
  partition = exp_x.sum(axis=1, keepdims=True)
  return exp_x / partition

# define cross entropy function
def cross_entropy(y_hat, y):
  return -nd.pick(y_hat, y).log()

# define trainer 
def train(params):
  for param in params:
    param[:] -= param.grad * LEARNING_RATE / BATCH_SIZE

# define forward function
def forward(x, W, b):
  return softmax(nd.dot(x.reshape((-1, INPUT_NUM)), W) + b)

# train
for i in range(ITERATION_COUNT):
  acc_loss = 0
  for feature, label in train_iter:
    with autograd.record():
      y_hat = forward(feature, W, b)
      l = cross_entropy(y_hat, label)
    l.backward()
    train([W, b])
    acc_loss += l.sum()
  print(f"aiteration {i+1} has acc loss: {acc_loss}")

print(f"result parameter: W: {W}, b: {b}")

# get test loss
acc_loss = 0
for X, y in test_iter:
  acc_loss += cross_entropy(forward(X, W, b), y)

print(f"average loss on test data: {acc_loss.mean()}")