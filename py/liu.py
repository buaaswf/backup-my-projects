﻿#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import sys
from sklearn.metrics import classification_report
sys.path.insert(0,"/home/s.li/2017/gpu4/caffe-segnet-cudnn5/python")
import matplotlib.pyplot as plt
import caffe
import os
import scipy.io
import shutil
from  mnist_single_plot_roc import drawroc
# Make sure that caffe is on the python path:
from sklearn.metrics import confusion_matrix
from tsne.tsne_1 import tsnepng

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
    caffe_root = "/home/s.li/2017/gpu4/caffe-segnet-cudnn5/"# this file is expected to be in {caffe_root}/examples
    sys.path.insert(0, caffe_root + 'python')
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    params = {'legend.fontsize':20}
    plt.rcParams.update(params)
    # plt.rcParams['image.cmap'] = 'gray'
    model =modelpath
    if not os.path.isfile(model):
        print("Downloading pre-trained CaffeNet model...")
    caffe.set_mode_cpu()
    net = caffe.Net(deployroot,model,caffe.TEST)
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    #  transformer.set_mean('data', np.load(caffe_root + meanroot).mean(1).mean(1))   # mean pixel
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( meanroot , 'rb' ).read()
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
    # lines =  labelfile(labelfilepath)
    #  print lines
    labels = []
    pred = []
    predroc= []
    nn = 0
    caffe.set_mode_gpu()
    allimages= GetFileList(inputimagedir, [])
    testimages =allimages
    # from random import shuffle
    import random
    # print allimages
    random.shuffle(testimages)
    errorimagelist="./error/mnist_result/"+outdir.split(".")[0]
    if not os.path.exists(errorimagelist):
        os.makedirs(errorimagelist)
    # print testimages

    for image in testimages:
        # print image,
        gtlabel = int(image.split("/")[-2])
        # print gtlabel
        try:
            net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image))
        except Exception, e:
            print nn
            print str(e)
            nn += 1
            continue
        out = net.forward()
        # pred.append(str(out['prob'].argmax()))
        #  print (out['prob'].shape)
        #  pred.append(out['prob'][1])
        # print("image is {}Predicted class is #{}.".format(image,out['prob'].argmax()))
        if out['prob'].argmax()!=gtlabel:
           print out['prob'].argmax(),gtlabel
           shutil.copy(image,errorimagelist+"/"+image.split("/")[-1].split(".")[0]+"_pred_"+str(out['prob'].argmax())+".png")
        # caffe.set_mode_gpu()
        # caffe.set_device(0)
        #net.forward()  # call once for allocation
        # %timeit net.forward()
        # feat = net.blobs[layername].data[1]
        feat = net.blobs[net.blobs.keys()[-2]].data[0]
        # for layer_name, param in net.params.iteritems():
                # print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
        # print net.blobs.keys()
        # filters = net.params['conv1'][0].data
        # print filters
        predroc.append(net.blobs[net.blobs.keys()[-1]].data[0].flatten())
        pred.append(np.argmax(net.blobs[net.blobs.keys()[-1]].data[0].flatten()))
        # print "===>>",net.blobs[net.blobs.keys()[-1]].data[0].flatten()
        # pred.append(out['prob'])
        # print out['prob']
        # print net.blobs[net.blobs.keys()[-2]].data[0]
        #np.savetxt(image+'feature.txt', feat.flat)
        #print type(feat.flat)
        featline = feat.flatten()
        # print featline
        #print type(featline)
        #featlinet= zip(*(featline))
        mat.append(featline)
        label=image.split("/")[-2]
        # labels.append(str(lines[nn][1]))
        labels.append(int(label))
        #  print "===>>",out['prob'].argmax()
        #  print "=====>>",lines[nn][1]
        if (nn%100==0):
            with open("./error/mnist_result/"+outdir,'w') as f:
                scipy.io.savemat(f, {'data' :mat,'labels':labels}) #append
        nn += 1

    # print pred.shape
    # tsnepng(mat,labels,"gootsne_"+outdir)
    print "tsnepng=========================>>>>"
	
    drawroc(labels,predroc,"./error/mnist_result/"+"zoomroc_10"+outdir.split('.')[0]+".png")
    print "roc=========================>>>>"
    print (classification_report(labels,pred))
    text_file = open("./error/mnist_result/"+outdir.split('.')[0]+".txt", "w")
    text_file.write(outdir.split('.')[0]+" model\n")
    text_file.write(classification_report(labels,pred))
    import pickle
    with open("./error/mnist_result/"+outdir.split('.')[0]+"_pred.pkl","wb") as f:
        pickle.dump(mat,f)
    with open("./error/mnist_result/"+outdir.split('.')[0]+"_true.pkl","wb") as f:
        pickle.dump(labels,f)
    with open("./error/mnist_result/"+outdir,'w') as f:
        scipy.io.savemat(f, {'data' :mat,'labels':labels}) #append
    cm=confusion_matrix(pred, labels)
    with open("./error/mnist_result/"+outdir.split(".")[0]+".pkl","wb") as f:
        pickle.dump(cm,f)
        print cm
        np.savetxt("./error/mnist_result/"+outdir.split(".")[0]+"mfse"+".csv", cm, delimiter=",")
def batch_extrac_featuretomat():
    #alexnet
    # alexnetpath="/home/swf/caffe/analysisfeatures/oversample/cifar10/cifar10_alex/"
    # alexnetpath="/home/swf/caffe/analysisfeatures/oversample/cifar10/cifar10_alex/"
    alexnetpath="/home/s.li/2016/caffe1128/caffe-master/models/"
    # googlenetpath="/home/swf/caffe/analysisfeatures/oversample/cifar10/cifar10_googlenet/"
    # cifar10netpath="/home/swf/caffe/analysisfeatures/oversample/cifar10/cifar_cifar10/"
    # svhn_cifar10netpath="/home/swf/caffe/analysisfeatures/oversample/svhn/cifar10net/"
    # svhn_googlenetpath="/home/swf/caffe/analysisfeatures/oversample/svhn/googlenet/"
    # svhn_alexnetpath="/home/swf/caffe/analysisfeatures/oversample/svhn/alexnet/"


              # googlenetpath+"bvlc_googlenet_iter_520000.caffemodel",\
              # googlenetpath+"oribvlc_googlenet_iter_520000.caffemodel",\
    # modelist=[alexnetpath+"oriciafr10caffe_alexnet_train_iter_390000.caffemodel",\
              # alexnetpath+"dvnciafr10caffe_alexnet_train_iter_450000.caffemodel",\
    # modelist=[alexnetpath + "cifar10gen_caffe_alexnet_train_iter_130000.caffemodel",\
    # modelist =[alexnetpath + "cifar10balanced0_caffe_alexnet_train_iter_410000.caffemodel",\
    # modelist =[alexnetpath + "7caffe_alexnet_train_iter_30000.caffemodel",\
    # modelist =[alexnetpath + "0509dvn/caffe_alexnet_train_iter_120000.caffemodel",\
    # modelist =[alexnetpath + "10caffe_alexnet_train_iter_10000.caffemodel",\
    modelist =[alexnetpath + "mnist/mnist_data/result1/caffe_alexnet_train_iter_140000.caffemodel",\
               alexnetpath + "mnist/mnist_data/result2/caffe_alexnet_train_iter_120000.caffemodel",\
               #alexnetpath + "mnist/mnist_data/result3/caffe_alexnet_train_iter_120000.caffemodel",\
               #alexnetpath + "mnist/mnist_data/result4/caffe_alexnet_train_iter_120000.caffemodel",\
               alexnetpath + "mnist/mnist_data/result5/caffe_alexnet_train_iter_120000.caffemodel",\
               alexnetpath + "mnist/mnist_data/result6/caffe_alexnet_train_iter_120000.caffemodel",\
               alexnetpath + "mnist/mnist_data/result7/caffe_alexnet_train_iter_50000.caffemodel",\
               #alexnetpath + "mnist/mnist_data/result8/caffe_alexnet_train_iter_60000.caffemodel",\
               alexnetpath + "mnist/mnist_data/result9/caffe_alexnet_train_iter_110000.caffemodel",\
               alexnetpath + "mnist/mnist_data/result10/caffe_alexnet_train_iter_50000.caffemodel",\
               alexnetpath + "mnist/mnist_data/result11/caffe_alexnet_train_iter_50000.caffemodel",\
               ]
    datalist=["/home/s.li/2017/gpu4/analysisfeatures/mnist_test/",
              "/home/s.li/2017/gpu4/analysisfeatures/mnist_test/",
              #"/home/s.li/2017/gpu4/analysisfeatures/mnist_test/",
              #"/home/s.li/2017/gpu4/analysisfeatures/mnist_test/",
              "/home/s.li/2017/gpu4/analysisfeatures/mnist_test/",
              "/home/s.li/2017/gpu4/analysisfeatures/mnist_test/",
              "/home/s.li/2017/gpu4/analysisfeatures/mnist_test/",
              #"/home/s.li/2017/gpu4/analysisfeatures/mnist_test/",
              "/home/s.li/2017/gpu4/analysisfeatures/mnist_test/",
              "/home/s.li/2017/gpu4/analysisfeatures/mnist_test/",
              "/home/s.li/2017/gpu4/analysisfeatures/mnist_test/",
              ]
    deploylist=[alexnetpath+"bvlc_alexnet/deploy.prototxt",
                alexnetpath+"bvlc_alexnet/deploy.prototxt",
                #alexnetpath+"bvlc_alexnet/deploy.prototxt",
                #alexnetpath+"bvlc_alexnet/deploy.prototxt",
                alexnetpath+"bvlc_alexnet/deploy.prototxt",
                alexnetpath+"bvlc_alexnet/deploy.prototxt",
                alexnetpath+"bvlc_alexnet/deploy.prototxt",
                #alexnetpath+"bvlc_alexnet/deploy.prototxt",
                alexnetpath+"bvlc_alexnet/deploy.prototxt",
                alexnetpath+"bvlc_alexnet/deploy.prototxt",
                alexnetpath+"bvlc_alexnet/deploy.prototxt",
                ]

    # meanlist=[alexnetpath+"patchcifa10_256_mean.binaryproto",
    meanlist=[alexnetpath+"mnist/mnist_data/mean/mean1.binaryproto",
              alexnetpath+"mnist/mnist_data/mean/mean2.binaryproto",
              #alexnetpath+"mnist/mnist_data/mean/mean3.binaryproto",
              #alexnetpath+"mnist/mnist_data/mean/mean4.binaryproto",
              alexnetpath+"mnist/mnist_data/mean/mean5.binaryproto",
              alexnetpath+"mnist/mnist_data/mean/mean6.binaryproto",
              alexnetpath+"mnist/mnist_data/mean/mean7.binaryproto",
              #alexnetpath+"mnist/mnist_data/mean/mean8.binaryproto",
              alexnetpath+"mnist/mnist_data/mean/mean9.binaryproto",
              alexnetpath+"mnist/mnist_data/mean/mean10.binaryproto",
              alexnetpath+"mnist/mnist_data/mean/mean11.binaryproto",
              ]
    shapelists=[[10,3,227,227],[10,3,227,227],\
                [10,3,227,227],[10,3,227,227],\
                [10,3,227,227],[10,3,227,227],\
                [10,3,227,227],[10,3,227,227]]#
                # [32,3,224,224],[32,3,224,224],\
                # [64,3,32,32],[64,3,32,32],[64,3,32,32],[64,3,32,32],
                # [10,3,227,227],[10,3,227,227],[10,3,224,224],[10,3,224,224]]
    labellist=["cifa10_valdst.csv",
               "cifa10_valdst.csv",
               "cifa10_valdst.csv",
               "cifa10_valdst.csv",
               "cifa10_valdst.csv",
               "cifa10_valdst.csv",
               "cifa10_valdst.csv",
               "cifa10_valdst.csv",]
               #"cifa10_valdst.csv",]
               # "svhn_ori_valdst.csv",]
    # labellist=[""]

    # outlist=["cifar10_alex_oversmaple.mat","cifar10_alex_ori.mat","cifar10_cifar10_dvn.mat",
             # "cifar10_cifar10_ori.mat","cifar10_google_dvn.mat","cifar10_google_ori.mat",
             # "svhn_cifar10_dvn.mat","svhn_cifar10_ori.mat","svhn_alex_dvn.mat","svhn_alex_ori.mat",
             # "svhn_google_dvn.mat","svhn_google_ori.mat"
             # ]
    outlist=["alex_mfseoverh1.mat","alex_mfseoverh2.mat",
	         "alex_mfseoverh5.mat","alex_mfseoverh6.mat",
             "alex_mfseoverh7.mat","alex_mfseoverh9.mat",
             "alex_mfseoverh10.mat","alex_mfseoverh11.mat",]
#]
    layernamelist=["fc8","fc8","fc8","fc8","fc8","fc8","fc8","fc8"]
                   # "ip1","ip1","fc8","fc8","loss3/classifier","loss3/classifier"]
    # layernamelist=["fc8","fc8","loss3/classifier","loss3/classifier","ip1","ip1",
                   # "ip1","ip1","fc8","fc8","loss3/classifier","loss3/classifier"]

    import traceback
    # for i in range(len(modelist)-1,len(modelist)):
    for i in range(0,len(modelist)):
    # for i in range(0,1):
        # if i<4 and i>1:
            # continue
    # for i in range(2,4):
        try:
          print modelist[i]
          net,transformer=loadmodel(modelpath=modelist[i],deployroot=deploylist[i],
                                    meanroot=meanlist[i],shapelist=shapelists[i])
          image2mat(net,transformer,datalist[i],outlist[i],labellist[i],layernamelist[i])
        except Exception as e:
          print e
          print traceback.format_exc()
          # break
          continue
        #argv[0] inputimagedir argv[1] labelfile



if __name__=='__main__':
    # if len(sys.argv)!=3:
        # print "Usage:python{}inputimagedir outdir labelfile".format(sys.argv[0])
    batch_extrac_featuretomat()
    #net,transformer=loadmodel(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
    #  net,transformer=loadmodel(modelpath='models/cifa10/cifar10_19layers_iter_200000.caffemodel',deployroot="models/cifa10/cifar10_deploy.prototxt",meanroot="data/cifar10-gcn-leveldb-splits/paddedmean.npy",shapelist=[100,3,32,32])
    #  #  net,transformer=loadmodel(modelpath='models/cifa10/cifar10_19layers_iter_200000.caffemodel',deployroot="models/scene/deploy.prototxt",shapelist=[50,3,100,100])
    #  image2mat(net,transformer,sys.argv[1],sys.argv[2],sys.argv[3])#argv[0] inputimagedir argv[1] labelfile

#def loadmodel(cafferoot,modelpath,deployroot,meanroot,shapelist=[64,3,100,100]):
