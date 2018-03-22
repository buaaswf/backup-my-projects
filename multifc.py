#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import sys
import matplotlib.pyplot as plt
import caffe
import os
import scipy.io
# Make sure that caffe is on the python path:

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
def GetFileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir.decode('gbk'))
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            #如果需要忽略某些文件夹，使用以下代码
            if s.endswith(".txt") or s.endswith(".sh") or s.endswith(".py"):
                continue
            #if int(s)>998 and int(s) < 1000:
            newDir=os.path.join(dir,s)
            GetFileList(newDir, fileList)
    return fileList
import csv
#dir = '/home/s.li/caffe0719/caffe-master/data/face/patch/casia1000/fullpathval.txt'
dir ='/home/s.li/data/align/Y_0/label_Y.txt'
def labelfile(dir,lines):
    with open (dir,'r') as f:
        lines = [line.strip().split(' ') for line in f ]
    #paths = [line[0] for line in lines]
    #labels = [line[1] for line in lines]
    print lines
    return lines





if len(sys.argv) != 3:
    print "Usage: python multifc.py inputimagedir feature.mat "
    sys.exit()

caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')


plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
model = 'models/casiaface/casia.caffemodel'
if not os.path.isfile(caffe_root + model):
    print("Downloading pre-trained CaffeNet model...")
#            !../scripts/download_model_binary.py ../models/bvlc_reference_caffenet}
caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'models/casiaface/casia_train_deploy.prototxt',
        caffe_root + model,
        caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.load(caffe_root + 'data/idface/casia_web.npy').mean(1).mean(1))   # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
# set net to batch size of 50
net.blobs['data'].reshape(64, 3, 100, 100)

inputimagedir = sys.argv[1]
mat = []
count = 0

for image in GetFileList(sys.argv[1], []):
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image))
    out = net.forward()
    print("Predicted class is #{}.".format(out['prob'].argmax()))
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net.forward()  # call once for allocation
    # %timeit net.forward()
    ifeat = net.blobs['ip3'].data[0]
    '''
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
    '''
    feat = net.blobs['ip3'].data[0]
    #np.savetxt(image+'feature.txt', feat.flat)
    #print type(feat.flat)
    featline = feat.flatten()
    #print type(featline)
    #featlinet= zip(*(featline))
    mat.append(featline)


with open('res.mat','w') as f:
        scipy.io.savemat(f, {'data' :mat}) #append

lines = labelfile(dir, [] )
labels = [ line[1] for line in lines]

with open('res.mat','ab') as f:
  scipy.io.savemat(f,{'labels':labels},oned_as='column')















