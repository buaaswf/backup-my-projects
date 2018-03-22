#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import os
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
net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(caffe_root + 'data/idface/val/5/flip41211_big.jpg'))
out = net.forward()
print("Predicted class is #{}.".format(out['prob'].argmax()))
caffe.set_device(0)
caffe.set_mode_gpu()
net.forward()  # call once for allocation
# %timeit net.forward()
feat = net.blobs['fc6'].data[0]
np.savetxt("feat.txt", feat.flat)
