#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import sys
import matplotlib.pyplot as plt
import caffe
import os
import scipy.io
from sklearn.externals import joblib
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


def labelfile(dir):
    lines = []
    with open (dir,'r') as f:
        lines = [line.strip().split(' ') for line in f ]
    #paths = [line[0] for line in lines]
    #labels = [line[1] for line in lines]
   # print lines
    return lines


def load_model():
    caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
    sys.path.insert(0, caffe_root + 'python')
    # plt.rcParams['figure.figsize'] = (10, 10)
    # plt.rcParams['image.interpolation'] = 'nearest'
    # plt.rcParams['image.cmap'] = 'gray'
    #model = 'models/casiaface/casia.caffemodel'
    model = 'models/casiaface/alignmixdata_iter_400000.caffemodel'
    #model = 'models/generateface/order_iter_330000.caffemodel'
    #model = 'models/casiaface/deepid'

    if not os.path.isfile(caffe_root + model):
        print("Downloading pre-trained CaffeNet model...")
    caffe.set_mode_gpu()
    net = caffe.Net(caffe_root + 'models/casiaface/casia_train_deploy.prototxt',
            caffe_root + model,
            caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.load(caffe_root + 'data/idface/casia_web.npy').mean(1).mean(1))   # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
    net.blobs['data'].reshape(64, 3, 144, 144)
    return net, transformer

def load_models(modelpath = "models/idface/",deploypath = "models/",meanfacepath =
"data/idface/",shapelist=[63,3,55,55]):

    caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
    sys.path.insert(0, caffe_root + 'python')
    # plt.rcParams['figure.figsize'] = (10, 10)
    # plt.rcParams['image.interpolation'] = 'nearest'
    # plt.rcParams['image.cmap'] = 'gray'
    model = modelpath # caffemodelroot
    if not os.path.isfile(caffe_root + model):
        print("Downloading pre-trained CaffeNet model...")
    caffe.set_mode_cpu()
    net = caffe.Net(caffe_root + deploypath, caffe_root + model,caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.load(caffe_root + meanfacepath).mean(1).mean(1))   # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
    net.blobs['data'].reshape(shapelist[0],shapelist[1],shapelist[2],shapelist[3])
    return net, transformer

def image2mat(image1, image2, outdir, net, transformer):
    mat = []
    nn = 0
    for image in [image1, image2]:
            try:
                net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image))
            except Exception, e:
                print nn
                print str(e)
                nn += 1
                continue
            #  out = net.forward()
            #  print("Predicted class is #{}.".format(out['prob'].argmax()))
            caffe.set_device(0)
            caffe.set_mode_gpu()
            out = net.forward()  # call once for allocation
            print("Predicted class is #{}.".format(out['prob'].argmax()))
            feat = net.blobs['ip3'].data[0]
            featline = feat.flatten()
            mat.append(featline)
            nn += 1
    print outdir
    with open(outdir,'w') as f:
        scipy.io.savemat(f, {'data' :mat}) #append

def image2mat_pca(image, outdir, net, transformer,ifreduce):
    print outdir
    mat = []
    nn = 0
    #for image in [image1, image2]:
    try:
       net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image))
    except Exception, e:
       print str(e)
    #  out = net.forward()
    #  print("Predicted class is #{}.".format(out['prob'].argmax()))
    net.forward()  # call once for allocation
    feat = net.blobs['ip3'].data[0]#"ip3"
    featline = feat.flatten()
    if (ifreduce == '1'):
        pca = joblib.load("/home/s.li/caffedeep/caffe-master/examples/pretrained/8model.m")
        featline = pca.transform(featline)
        print "pca done"
    mat.append(featline)
    with open(outdir,'w') as f:
        scipy.io.savemat(f, {'data' :mat}) #append
