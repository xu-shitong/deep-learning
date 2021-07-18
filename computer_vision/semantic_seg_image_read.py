import d2lzh 
from mxnet import nd

colormap2label = nd.zeros(256 ** 3)
for i, pixel in enumerate(d2lzh.VOC_COLORMAP):
  colormap2label[(pixel[0] * 256 + pixel[1]) * 256 + pixel[2]] = i

voc_dir = '../voc-2012/VOCdevkit/VOC2012'

train_feature, train_labels = d2lzh.read_voc_images(voc_dir)

# reading one picture
x = 200
y = 196
img = train_labels[0]
# fig = d2lzh.plt.imshow(img.asnumpy())

# # mark the pixel being examined
# mark = d2lzh.plt.Rectangle(xy=(x, y), width=2, height=2, edgecolor='blue', linewidth=2)
# fig.axes.add_patch(mark)

# # get pixel info and lookup label
# img_matrix = img.astype('int32')
# pixel_index = (img_matrix[y, x, 0] * 256 + img_matrix[y, x, 1]) * 256 + img_matrix[y, x, 2]
# print(colormap2label[pixel_index])
# print(d2lzh.VOC_CLASSES[int(colormap2label[pixel_index.asscalar()].asscalar())])

# preprocess one image, cut in different sizes
imgs = [[], []]
for i in range(0,5):
  feature, label = d2lzh.voc_rand_crop(train_feature[0], train_labels[0], 225, 225)
  imgs[0].append(feature)
  imgs[1].append(label)
d2lzh.show_images(imgs[0] + imgs[1], 1, 5)

d2lzh.plt.show()
