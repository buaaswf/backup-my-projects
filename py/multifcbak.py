#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import sys
import matplotlib.pyplot as plt
import caffe
import os
import re
# Make sure that caffe is on the python path:

def GetFileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir.decode('gbk'))
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            #如果需要忽略某些文件夹，使用以下代码
            #if s == "xxx":
                #continue
            newDir=os.path.join(dir,s)
            GetFileList(newDir, fileList)
    return fileList
if len(sys.argv) != 3:
    print "Usage: python multifc.py inputimagedir feature.mat "
    sys.exit()

caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')


plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

if not os.path.isfile(caffe_root + 'models/idface/deepid_28_0_train_iter_80000.caffemodel'):
    print("Downloading pre-trained CaffeNet model...")
#            !../scripts/download_model_binary.py ../models/bvlc_reference_caffenet}
caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'models/idface/deploy2.prototxt',
        caffe_root + 'models/idface/deepid_28_0_train_iter_80000.caffemodel',
        caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.load(caffe_root + 'data/idface/idface.npy').mean(1).mean(1))   # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
# set net to batch size of 50
net.blobs['data'].reshape(50, 3, 55, 55)

def vis_square(resname, data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imsave(resname, data)
inputimagedir = sys.argv[1]
for image in GetFileList(sys.argv[1],[]):
    print image
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image))
    filename = re.compile('\/{1}_(.*?)_\.jpg')
    filename.findall(image)
    out = net.forward()
    print("Predicted class is #{}.".format(out['prob'].argmax()))
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net.forward()  # call once for allocation
    # %timeit net.forward()
    feat = net.blobs['fc6'].data[0]
    np.savetxt("feat.txt", feat.flat)
    # the parameters are a list of [weights, biases]
    filters = net.params['conv1'][0].data
    vis_square("conv1.jpg", filters.transpose(0, 2, 3, 1))
    feat = net.blobs['conv1'].data[0, :36]
    vis_square("feat1.jpg", feat, padval=1)
    filters = net.params['conv2'][0].data
    # vis_square("conv2.jpg", filters[:48].reshape(48 ** 2, 5, 5))
    feat = net.blobs['conv2'].data[0, :36]
    vis_square("feat2.jpg", feat, padval=1)
    feat = net.blobs['conv3'].data[0]
    vis_square("feat3.jpg", feat, padval=0.5)
    feat = net.blobs['conv4'].data[0]
    vis_square(image+"feat4.jpg", feat, padval=0.5)
    feat = net.blobs['fc6'].data[0]
    np.savetxt(image+'feature.txt', feat.flat)










