from numpy.core.fromnumeric import shape
from mxnet import autograd, nd 
from d2lzh import plt

FEATURE_NUM = 2
SAMPLE_NUM = 1000
true_w = nd.array([2, -3.4])
true_b = 4.2

# generate training dataset
features = nd.random.normal(0, 1, shape=(SAMPLE_NUM, FEATURE_NUM))
labels = nd.dot(features, true_w.T)
labels = labels + true_b
labels += nd.random.normal(0, 1, shape=labels.shape)

# parameters to be trained
w = nd.random.normal(0, 1, shape=(FEATURE_NUM, 1))
d = nd.zeros(shape=(1,))
w.attach_grad()
d.attach_grad()

# training using square loss function
def forward(X):
  return nd.dot(X, w) + d

def square_loss(labels, hat_labels):
  return (labels - hat_labels.reshape(labels.shape)) ** 2 / 2

# calculate new parameter after one iteration
def sgd(params):
  for param in params:
    param[:] = param - LEARNING_RATE * param.grad / SAMPLE_NUM

REGRESSION_NUM = 3 # iteration of regression time
LEARNING_RATE = 1 
for i in range(REGRESSION_NUM):

  hat_labels = forward(features)
  with autograd.record():
    loss = square_loss(labels, hat_labels)
  loss.backward()

  sgd([w, d])
  print(f"epoch {i}, loss {square_loss(forward(features), labels)}")

print(w, d)
