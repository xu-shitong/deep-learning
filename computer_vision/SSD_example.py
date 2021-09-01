from mxnet import nd, contrib
from mxnet.gluon import nn

def flatten_pred(pred):
  return pred.transpose((0, 2, 3, 1)).flatten()
def concat_preds(preds):
  return nd.concat(*[flatten_pred(p) for p in preds], dim=1)


def cls_predictor(num_anchors, num_classes):
  return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3,padding=1)

def bbox_predictor(num_anchors):
  return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor): 
  Y = blk(X)
  anchors = contrib.ndarray.MultiBoxPrior(Y, sizes=size, ratios=ratio) 
  cls_preds = cls_predictor(Y)
  bbox_preds = bbox_predictor(Y)
  return (Y, anchors, cls_preds, bbox_preds)

def base_net():
  blk = nn.Sequential()
  for num_filters in [16, 32, 64]: 
    blk.add(down_sample_blk(num_filters))
  return blk

def down_sample_blk(num_channels): 
  blk = nn.Sequential()
  for _ in range(2):
    blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                            nn.BatchNorm(in_channels=num_channels),
                            nn.Activation('relu'))
  blk.add(nn.MaxPool2D(2))
  return blk

def get_blk(i): 
  if i == 0:
    blk = base_net() 
  elif i == 4:
    blk = nn.GlobalMaxPool2D() 
  else:
    blk = down_sample_blk(128)
  return blk


sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
          [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1



class TinySSD(nn.Block):
  def __init__(self, num_classes, **kwargs):
    super(TinySSD, self).__init__(**kwargs) 
    self.num_classes = num_classes
    for i in range(5):
      # 即赋值语句self.blk_i = get_blk(i)
      setattr(self, 'blk_%d' % i, get_blk(i))
      setattr(self, 'cls_%d' % i, cls_predictor(num_anchors, num_classes)) 
      setattr(self, 'bbox_%d' % i, bbox_predictor(num_anchors))
  def forward(self, X):
    anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5 
    for i in range(5):
      # getattr(self, 'blk_%d' % i)即访问self.blk_i
      X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
        X, getattr(self, 'blk_%d' % i), sizes[i], ratios[i],
        getattr(self, 'cls_%d' % i), getattr(self, 'bbox_%d' % i)) # reshape函数中的0表示保持批量大小不变
      # print(cls_preds[i].shape)
      # print(bbox_preds[i].shape)
    return (nd.concat(*anchors, dim=1), 
            concat_preds(cls_preds).reshape((0, -1, self.num_classes + 1)), 
            concat_preds(bbox_preds)
           )

# def forward(x, block): 
#   block.initialize()
#   return block(x)

# Y1 = forward(nd.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
# Y2 = forward(nd.zeros((2, 16, 10, 10)), cls_predictor(3, 10))
# print(Y1.shape, Y2.shape)

# print(concat_preds([Y1, Y2]).shape)
# print(forward(nd.zeros((2, 3, 20, 20)), down_sample_blk(10)).shape)

# print(forward(nd.zeros((2, 3, 256, 256)), base_net()).shape)

net = TinySSD(num_classes=1)
net.initialize()
X = nd.ones((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)


# print('output anchors:', anchors.shape) 
# print('output class preds:', cls_preds.shape) 
# print('output bbox preds:', bbox_preds.shape)
# print(cls_preds[0, 10, :])
print(f"cls_preds = {cls_preds}")
print(f"bbox_pred = {bbox_preds}")