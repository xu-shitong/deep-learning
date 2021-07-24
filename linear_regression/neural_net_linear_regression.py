import mxnet 
from mxnet import nd, autograd, init
from mxnet.gluon import data as gdata, loss as gloss, nn
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
LEARNING_RATE = 0.03
net = nn.Dense(1)
net.initialize(init=init.Normal(sigma=0.01))

# training
dataset = gdata.ArrayDataset(features_train, labels_train)
data_iter = gdata.DataLoader(dataset, BATCH_SIZE, shuffle=True)
loss = gloss.L2Loss()
trainer = mxnet.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': LEARNING_RATE})
for i in range(ITERATION_COUNT):
  acc_loss = 0
  for feature, label in data_iter:
    with autograd.record():
      y_hat = net(feature)
      l = loss(y_hat, label)
    l.backward()

    trainer.step(BATCH_SIZE)
    acc_loss += l.sum()
  print(f"iteration {i+1} accumulated loss: {acc_loss}")

print(f"training result: w: {net.weight.data()}, d: {net.bias.data()}")

# get loss on test data 
l = loss(net(features_test), labels_test)
print(f"average loss on test data: {l.mean()}")
