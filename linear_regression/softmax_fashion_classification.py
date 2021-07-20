from mxnet.gluon import data as gdata
import d2lzh

mnist_train = gdata.vision.FashionMNIST(train=True)
features, labels = mnist_train[:5]

d2lzh.show_fashion_mnist(features, d2lzh.get_fashion_mnist_labels(labels))
d2lzh.plt.show()