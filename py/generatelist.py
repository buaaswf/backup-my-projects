#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import sys
#import matplotlib.pyplot as plt
import os
import caffe
import math
def generatelist(listtxt,imagepath):
    list1=[]
    list2=[]
    labellist=[]
    content = []
    with open(listtxt) as f:
        content = f.readlines()
        print content[0]
    count = 0
    for image in content:
        if count == 0:
           count += 1
           continue
        if count < int(content[0])+1:
            #print count
            labellist.append(1)
            count += 1
            path = image.split()
            list1.append(imagepath+path[0]+'/'+path[0]+'_'+format(int(path[1]),'04d')+'_aligned.jpg')
            list2.append(imagepath+path[0]+'/'+path[0]+'_'+format(int(path[2]),'04d')+'_aligned.jpg')
            #  if int(path[1]) >= 10:
                #  print path[1]
                #  list1.append(imagepath+path[0]+'/'+path[0]+'_'+'00'+path[1]+'_aligned.jpg')
            #  else:
                #  list1.append(imagepath+path[0]+'/'+path[0]+'_'+'000'+path[1]+'_aligned.jpg')
            #  if int(path[2]) < 10:
                #  list2.append(imagepath+path[0]+'/'+path[0]+'_'+'000'+path[2]+'_aligned.jpg')
            #  else:
                #  list2.append(imagepath+path[0]+'/'+path[0]+'_'+'00'+path[2]+'_aligned.jpg')
        else:
            count > int(content[0])
            labellist.append(0)
            count += 1
            path = image.split()
            print path
            list1.append(imagepath+path[0]+'/'+path[0]+'_'+format(int(path[1]),'04d')+'_aligned.jpg')
            list2.append(imagepath+path[2]+'/'+path[2]+'_'+format(int(path[3]),'04d')+'_aligned.jpg')
            #list1.append(imagepath+path[0]+'/'+path[1]+'_aligned.jpg')
            #  if int(path[1]) >= 10 and int (path[1]<100):
                #  list1.append(imagepath+path[0]+'/'+path[0]+'_'+format(path[1],'04d')+'_aligned.jpg')
            #  else if len(path[1])>=3:
                #  list1.append(imagepath+path[0]+'/'+path[0]+'_'+'000'+path[1]+'_aligned.jpg')
            #  if int(path[3]) < 10:
                #  list2.append(imagepath+path[2]+'/'+path[2]+'_'+'000'+path[3]+'_aligned.jpg')
            #  else:
                #  list2.append(imagepath+path[2]+'/'+path[2]+'_'+'00'+path[3]+'_aligned.jpg')
            #  if int(path[3]) >= 100:
                #  list2.append(imagepath+path[2]+'/'+path[2]+'_'+'0'+path[3]+'_aligned.jpg')
            #  else:
                #  list2.append(imagepath+path[2]+'/'+path[2]+'_'+'00'+path[3]+'_aligned.jpg')
            #print list1
    return list1,  list2, labellist
def featuresiliar(list1, list2, labellist,net,transformer):
    similar =[]
    caffe.set_device(0)
    caffe.set_mode_gpu()
    regularnum = 0
    truenum = 0
    for image1,image2 in zip(list1,list2):
        mat = []
        regularnum += 1
        if os.path.exists(image1) and os.path.exists(image2): #and regularnum <= 500:
            if regularnum < 500:
                   truenum += 1
            for image in [image1,image2]:
                    try:
                        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image))
                    except Exception,e:
                        #  print nn
                        print str(e)
                        #  nn += 1
                        continue
                    net.forward()
                    feat=net.blobs['ip3'].data[0]
                    featline = feat.flatten()
                    mat.append(featline)
            cos = np.dot(mat[0],mat[1])/(math.sqrt(np.dot(mat[0],mat[0]))*math.sqrt(np.dot(mat[1],mat[1])))
            similar.append(cos)
            with open("similar.txt",'w') as file:
                for item in similar:
                    file.write("{}\n".format(item))
        else:
            continue
    return similar ,truenum
def accuracy(similar,t,truenum):
    count = 0
    accuracy = 0
    error=0
    for s in similar:
        count += 1
        if count < truenum:
            if s > t:
                accuracy += 1
            else:
                error += 1
        else:
            if  s <= t:
                accuracy += 1
            else:
                error +=1
    #  file.writelines( list( "%s\n" % item for item in similar ) )
    return accuracy,error
def load_model(caffepath='../',modelpath='models/casiaface/casia.caffemodel',deployroot='models/casiaface/casia_train_deploy.prototxt', meanroot='data/idface/casia_web.npy',shapelist=[64,3,100,100]):
    caffe_root = caffepath  # this file is expected to be in {caffe_root}/examples
    sys.path.insert(0, caffe_root + 'python')
    #plt.rcParams['figure.figsize'] = (10, 10)
    #  plt.rcParams['image.interpolation'] = 'nearest'
    #  plt.rcParams['image.cmap'] = 'gray'
    model =modelpath
    if not os.path.isfile(caffe_root + model):
        print("Downloading pre-trained CaffeNet model...")
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(caffe_root + deployroot,caffe_root + model,caffe.TEST)
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.load(caffe_root + meanroot).mean(1).mean(1))   # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
# set net to batch size of 50
    net.blobs['data'].reshape(shapelist[0], shapelist[1], shapelist[2], shapelist[3])
    return net,transformer

def list2accuracy():
    list1,list2,labellist = generatelist('./generatemat/pairsDevTest.txt','./generatemat/data/alignlfw_funneled/')
    net,transformer = load_model()
    similar,truenum=featuresiliar(list1,list2,labellist,net,transformer)
    #  print similar
    print truenum
    acclist= []
    for i in range(0,100,1):
        t=i*0.01
        accuracyres,error=accuracy(similar,t,truenum)
        acclist.append([i,accuracyres,error])
    print acclist

if __name__=='__main__':
    list2accuracy()







