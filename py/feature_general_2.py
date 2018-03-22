#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import sys
sys.path.insert(0,"/home/swf/caffe/python/")
import matplotlib.pyplot as plt
import caffe
import os
import scipy.io
import shutil
from single_plot_roc import drawroc
# Make sure that caffe is on the python path:
from sklearn.metrics import confusion_matrix

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
#dir = '/home/s.li/caffe0719/caffe-master/data/face/patch/casia1000/fullpathval.txt'
def labelfile(dir):
    lines = []
    with open (dir,'r') as f:
        lines = [line.strip().split(' ') for line in f ]
    #paths = [line[0] for line in lines]
    #labels = [line[1] for line in lines]
   # print lines
    return lines

if len(sys.argv) != 4:
    print "Usage: python multifc.py inputimagedir feature.mat labeldir"
    #  sys.exit()

def loadmodel(caffepath='../',modelpath='models/casiaface/casia.caffemodel',deployroot='models/casiaface/casia_train_deploy.prototxt',meanroot='data/idface/casia_web.npy',shapelist=[64,3,100,100]):
    #  caffe_root = caffepath  # this file is expected to be in {caffe_root}/examples
    caffe_root = ""# this file is expected to be in {caffe_root}/examples
    sys.path.insert(0, caffe_root + 'python')
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    model =modelpath
    if not os.path.isfile(caffe_root + model):
        print("Downloading pre-trained CaffeNet model...")
    caffe.set_mode_cpu()
    net = caffe.Net(deployroot,caffe_root + model,caffe.TEST)
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    #  transformer.set_mean('data', np.load(caffe_root + meanroot).mean(1).mean(1))   # mean pixel
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( caffe_root+meanroot , 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    out = arr[0]


    transformer.set_mean('data', out.mean(1).mean(1))   # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
# set net to batch size of 50
    net.blobs['data'].reshape(shapelist[0], shapelist[1], shapelist[2], shapelist[3])
    return net,transformer

def image2mat(net,transformer,inputimagedir,outdir,labelfilepath,layername):
    #inputimagedir = sys.argv[1]
    mat = []
    lines =  labelfile(labelfilepath)
    #  print lines
    labels = []
    pred = []
    nn = 0
    for image in GetFileList(inputimagedir, []):
        #  print image
        try:
            net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image))
        except Exception, e:
            print nn
            print str(e)
            nn += 1
            continue
        out = net.forward()
        caffe.set_mode_gpu()
        pred.append(str(out['prob'].argmax()))
        #  print("image is {}Predicted class is #{}.".format(image,out['prob'].argmax()))
        if out['prob'].argmax()!=lines[nn][1]:
           shutil.copy(image,'./error')
        caffe.set_mode_gpu()
        caffe.set_device(0)
        #net.forward()  # call once for allocation
        # %timeit net.forward()
        feat = net.blobs[layername].data[0]
        #np.savetxt(image+'feature.txt', feat.flat)
        #print type(feat.flat)
        featline = feat.flatten()
        #print type(featline)
        #featlinet= zip(*(featline))
        mat.append(featline)
        labels.append(str(lines[nn][1]))
        #  print "===>>",out['prob'].argmax()
        #  print "=====>>",lines[nn][1]
        if (nn%100==0):
            with open(outdir,'w') as f:
                scipy.io.savemat(f, {'data' :mat,'labels':labels}) #append
        nn += 1
    #  print pred
    # drawroc(labels,pred,outdir.split('.')[0]+".png")
    from tsne.tsne_1 import tsnepng
    tsnepng(pred,labels,"tsne_"+outdir.split('.')[0]+".png")
    import pickle
    with open("pred.pkl","wb") as f:
        pickle.dump(pred,f)
    with open("true","wb") as f:
        pickle.dump(labels,f)
    with open(outdir,'w') as f:
        scipy.io.savemat(f, {'data' :mat,'labels':labels}) #append
    cm=confusion_matrix(pred, labels)
    with open(outdir.split(".")[0]+".pkl","wb") as f:
        pickle.dump(cm,f)
        print cm
def batch_extrac_featuretomat():
    #alexnet
    alexnetpath="/home/swf/caffe/analysisfeatures/dvns/cifar10/cifar10_alex/"
    googlenetpath="/home/swf/caffe/analysisfeatures/dvns/cifar10/cifar10_googlenet/"
    cifar10netpath="/home/swf/caffe/analysisfeatures/dvns/cifar10/cifar_cifar10/"
    svhn_cifar10netpath="/home/swf/caffe/analysisfeatures/dvns/svhn/cifar10net/"
    svhn_googlenetpath="/home/swf/caffe/analysisfeatures/dvns/svhn/googlenet/"
    svhn_alexnetpath="/home/swf/caffe/analysisfeatures/dvns/svhn/alexnet/"


    modelist=[alexnetpath+"dvnciafr10caffe_alexnet_train_iter_450000.caffemodel",\
              alexnetpath+"oriciafr10caffe_alexnet_train_iter_390000.caffemodel",\
              googlenetpath+"bvlc_googlenet_data_generate_iter_5120000.caffemodel",\
              googlenetpath+"bvlc_googlenet_data_ori_iter_5360000.caffemodel",\
              cifar10netpath+"svhn_combine_47_96_easy_iter_300000.caffemodel",\
              cifar10netpath+"svhn_ori_iter_60000.caffemodel.h5",\
              svhn_cifar10netpath+"svhn_combine_47_96_easy_iter_300000.caffemodel",\
              svhn_cifar10netpath+"svhn_ori_iter_60000.caffemodel.h5",\
              svhn_alexnetpath+"svhn_caffe_alexnet_train_iter_120000.caffemodel",\
              svhn_alexnetpath+"svhn_caffe_alexnet_train_iter_120000.caffemodel",\
              svhn_googlenetpath+"svhn_bvlc_googlenet_ge_iter_426467.caffemodel",\
              svhn_googlenetpath+"svhn_bvlc_googlenet_iter_110000.caffemodel"\


              ]
    datalist=["/home/swf/caffe/analysisfeatures/generatemat/data/cifa10_val/",
              "/home/swf/caffe/analysisfeatures/generatemat/data/cifa10_val/",
              "/home/swf/caffe/analysisfeatures/generatemat/data/cifa10_val/",
              "/home/swf/caffe/analysisfeatures/generatemat/data/cifa10_val/",
              "/home/swf/caffe/analysisfeatures/generatemat/data/cifa10_val/",
              "/home/swf/caffe/analysisfeatures/generatemat/data/cifa10_val/",
              "/home/swf/caffe/analysisfeatures/generatemat/data/svhn_ori_val/",
              "/home/swf/caffe/analysisfeatures/generatemat/data/svhn_ori_val/",
              "/home/swf/caffe/analysisfeatures/generatemat/data/svhn_ori_val/",
              "/home/swf/caffe/analysisfeatures/generatemat/data/svhn_ori_val/",
              "/home/swf/caffe/analysisfeatures/generatemat/data/svhn_ori_val/",
              "/home/swf/caffe/analysisfeatures/generatemat/data/svhn_ori_val/",
              ]
    deploylist=[alexnetpath+"deploy.prototxt",
                alexnetpath+"deploy.prototxt",
                cifar10netpath+"",
                cifar10netpath+"",
                googlenetpath+"",
                googlenetpath+"",
                svhn_cifar10netpath+"cifar10_full.prototxt",
                svhn_cifar10netpath+"cifar10_full.prototxt",
                svhn_alexnetpath+"deploy.prototxt",
                svhn_alexnetpath+"deploy.prototxt",
                svhn_googlenetpath+"deploy.prototxt",
                svhn_googlenetpath+"deploy.prototxt"
                ]
    meanlist=[alexnetpath+"patchcifa10_256_mean.binaryproto",
              alexnetpath+"patchcifa10_256_mean.binaryproto",
              alexnetpath+"patchcifa10_256_mean.binaryproto",
              alexnetpath+"patchcifa10_256_mean.binaryproto",
              alexnetpath+"mean.binaryproto",
              alexnetpath+"mean.binaryproto",
              svhn_cifar10netpath+"patchcombine_47_96_easy_256_mean.binaryproto",
              svhn_cifar10netpath+"patchcombine_47_96_easy_256_mean.binaryproto",
              svhn_cifar10netpath+"patchcombine_47_96_easy_256_mean.binaryproto",
              svhn_cifar10netpath+"patchcombine_47_96_easy_256_mean.binaryproto",
              svhn_cifar10netpath+"patchcombine_47_96_easy_256_mean.binaryproto",
              svhn_cifar10netpath+"patchcombine_47_96_easy_256_mean.binaryproto"
              ]
    shapelists=[[10,3,227,227],[10,3,227,227],[10,3,224,224],[10,3,224,224],
                [64,3,32,32],[64,3,32,32],[1,3,32,32],[1,3,32,32],
                [10,3,227,227],[10,3,227,227],[10,3,224,224],[10,3,224,224]]
    labellist=["cifa10_valdst.csv",
               "cifa10_valdst.csv",
               "cifa10_valdst.csv",
               "cifa10_valdst.csv",
               "cifa10_valdst.csv",
               "cifa10_valdst.csv",
               "svhn_ori_valdst.csv",
               "svhn_ori_valdst.csv",
               "svhn_ori_valdst.csv",
               "svhn_ori_valdst.csv",
               "svhn_ori_valdst.csv",
               "svhn_ori_valdst.csv"]
    outlist=["cifar10_alex_dvn.mat","cifar10_alex_ori.mat","cifar10_cifar10_dvn.mat",
             "cifar10_cifar10_ori.mat","cifar10_google_dvn.mat","cifar10_google_ori.mat",
             "svhn_cifar10_dvn.mat","svhn_cifar10_ori.mat","svhn_alex_dvn.mat","svhn_alex_ori.mat",
             "svhn_google_dvn.mat","svhn_google_ori.mat"
             ]
    layernamelist=["fc8","fc8","loss3/classifier","loss3/classifier","ip1","ip1",
                   "ip1","ip1","fc8","fc8","loss3/classifier","loss3/classifier"]
    for i in range(10,len(modelist)):
        print modelist[i]
        net,transformer=loadmodel(modelpath=modelist[i],deployroot=deploylist[i],
                                  meanroot=meanlist[i],shapelist=shapelists[i])
        image2mat(net,transformer,datalist[i],outlist[i],labellist[i],layernamelist[i])
        #argv[0] inputimagedir argv[1] labelfile



if __name__=='__main__':
    if len(sys.argv)!=3:
        print "Usage:python{}inputimagedir outdir labelfile".format(sys.argv[0])
    batch_extrac_featuretomat()
    #net,transformer=loadmodel(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
    #  net,transformer=loadmodel(modelpath='models/cifa10/cifar10_19layers_iter_200000.caffemodel',deployroot="models/cifa10/cifar10_deploy.prototxt",meanroot="data/cifar10-gcn-leveldb-splits/paddedmean.npy",shapelist=[100,3,32,32])
    #  #  net,transformer=loadmodel(modelpath='models/cifa10/cifar10_19layers_iter_200000.caffemodel',deployroot="models/scene/deploy.prototxt",shapelist=[50,3,100,100])
    #  image2mat(net,transformer,sys.argv[1],sys.argv[2],sys.argv[3])#argv[0] inputimagedir argv[1] labelfile

#def loadmodel(cafferoot,modelpath,deployroot,meanroot,shapelist=[64,3,100,100]):
