from mxnet import nd, autograd
from mxnet.gluon import data as gdata 
import d2lzh

# define true parameters
W = nd.array([3,4])
b = 0.5

# create training dataset
train_size = 1000
features_train = nd.random.normal(0, 1, shape=(train_size, W.shape[0]))
labels_train = nd.dot(features_train, W.T) + b
labels_train += nd.random.normal(0, 0.01, shape = labels_train.shape)

# create test dataset
test_size = 10
features_test = nd.random.normal(0, 1, shape=(test_size, W.shape[0]))
labels_test = nd.dot(features_test, W) + b 
labels_test += nd.random.normal(0, 0.01, shape = labels_test.shape)

# # plot features
# d2lzh.plt.scatter(features_train.T[0].asnumpy(), labels_train.T.asnumpy(), 1)
# d2lzh.plt.show()

# define training parameters
BATCH_SIZE = 10
ITERATION_COUNT = 10
LEARNING_RATE = 0.3
W_train = nd.random.normal(0, 1, shape=W.shape).reshape((W.shape[0], 1))
W_train.attach_grad()
b_train = nd.array([0])
b_train.attach_grad()

# define forward function
def forward(W, b, x):
  return nd.dot(x, W) + b

# define square loss function
def loss(y, y_hat):
  return nd.power(y - y_hat.reshape(y.shape), 2) / 2

# define parameter update function
def update(W_train, b_train):
  W_train[:] -= W_train.grad * LEARNING_RATE / BATCH_SIZE
  b_train[:] -= b_train.grad * LEARNING_RATE / BATCH_SIZE

# training
dataset = gdata.ArrayDataset(features_train, labels_train)
data_iter = gdata.DataLoader(dataset, BATCH_SIZE, shuffle=True)
for i in range(ITERATION_COUNT):

  acc_loss = 0
  for feature, label in data_iter:
    with autograd.record():
      y_hat = forward(W_train, b_train, feature)
      l = loss(label, y_hat)

    l.backward()
    update(W_train, b_train)

    acc_loss += l.sum()
  print(f"iteration {i+1} has acc loss: {acc_loss}")

print(f"final parameters are: W: {W_train}, b: {b_train}")

# get loss on test data
y_hat = loss(forward(W_train, b_train, features_test), labels_test)
print(f"average loss on test data: {y_hat.mean()}")
