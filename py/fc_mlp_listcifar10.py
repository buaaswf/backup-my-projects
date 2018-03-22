#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = '/home/user/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
#  plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.cmap'] = 'jet'

import os


#def GetFileList():
def GetFileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir.decode('gbk'))
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            #如果需要忽略某些文件夹，使用以下代码
            if s.endswith(".abs") or s.endswith(".sh") or s.endswith(".py") or s.endswith(".prototxt") or s.endswith(".solverstate"):
                continue
            #if int(s)>998 and int(s) < 1000:
            newDir=os.path.join(dir,s)
            GetFileList(newDir, fileList)
    return fileList

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
alexnetpath="./thyroid/"
#modelnet = 'deepid_aligned_Casia_webFace_iter_450000.caffemodel'
# modelnet = 'examples/cifar10/bvlc_googlenet_data_ori_iter_200000.caffemodel'
#modelnet= alexnetpath +'alexnet/train_val_3combines_pool5_pooling_iter_380000.caffemodel'
modelnet= alexnetpath +'alexnet/0601.caffemodel'
#  modelnet = 'examples/cifar10/bvlc_googlenet_data_generate_iter_5120000.caffemodel'
if not os.path.isfile(modelnet):
        print("Downloading pre-trained CaffeNet model...")
#            !../scripts/download_model_binary.py ../models/bvlc_reference_caffenet}
caffe.set_mode_gpu()
#net = caffe.Net(caffe_root + 'models/idface/deploy2.prototxt',
#                caffe_root + 'models/' + modelnet,
#                caffe.TEST)

# net = caffe.Net(alexnetpath+"alexnet/pool5.prototxt",
#net = caffe.Net(alexnetpath+"alexnet/conv5.prototxt",
net = caffe.Net(alexnetpath+"alexnet/deploy2.prototxt",
                modelnet,
                caffe.TEST)
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
blob = caffe.proto.caffe_pb2.BlobProto()
data = open(alexnetpath+"patchthyroid_mean_100.binaryproto", 'rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))
out = arr[0]

transformer.set_mean('data', out.mean(1).mean(1))  # mean pixel
# transformer.set_mean('data', np.load(alexnetpath+"patchthyroid_mean_100.binaryproto").mean(1).mean(1))   # mean pixel
# transformer.set_mean('data', np.load(caffe_root + 'examples/cifar10/mean.npy').mean(1).mean(1))   # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
# set net to batch size of 50
net.blobs['data'].reshape(32, 3, 227, 227)
#net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(caffe_root + 'data/idface/val/5/flip41211_big.jpg'))
count = 0
#import re
#pattern = re.cmpile()
for img in GetFileList(r"./thyroid/0715test/1",[]):
    name = img.split("/")[-1].split(".jpg")[0]
    count += 1
    print img,count
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img))
    out = net.forward()
    print("Predicted class is #{}.".format(out['prob'].argmax()))
#caffe.set_device(0)
#caffe.set_mode_gpu()
    net.forward()  # call once for allocation
# %timeit net.forward()
#feat = net.blobs['fc6'].data[0]
#np.savetxt("feat.txt", feat.flat)

    #dst=r"./featureimages/conv50_ubuntu/"
    dst=r"./featureimages/orialex1/"
    filters = net.params['conv1'][0].data
    print filters.shape
    vis_square(dst+name+str(count)+"conv1.jpg", filters.transpose(0, 2, 3, 1))
    feat = net.blobs['conv1'].data[0, :36]
    vis_square(dst+name+str(count)+"feat1.jpg", feat, padval=1)
    filters = net.params['conv2'][0].data
# vis_square("conv2.jpg", filters[:48].reshape(48 ** 2, 5, 5))
    feat = net.blobs['conv2'].data[0, :36]
    vis_square(dst+name+str(count)+"feat2.jpg", feat, padval=1)
    feat = net.blobs['conv3'].data[0]
    vis_square(dst+name+str(count)+"feat3.jpg", feat, padval=0.5)
    feat = net.blobs['conv4'].data[0]
    vis_square(dst+name+str(count)+"feat4.jpg", feat, padval=0.5)

    feat = net.blobs['conv5'].data[0]
    vis_square(dst+name+str(count)+"feat5.jpg", feat, padval=0.5)
#=================================================================
    #feat = net.blobs['inception_3a/output'].data[0]
    #feat = net.blobs['fc7'].data[0]
    #vis_square(dst+name+str(count)+"feat6.jpg", feat, padval=1)

    feat = net.blobs['fc8new'].data[0]
    np.savetxt(dst+name+str(count)+'feature.txt', feat.flat)








