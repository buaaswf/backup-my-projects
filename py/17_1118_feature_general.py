#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import sys
sys.path.insert(0,"/home/user/caffe/python")
import matplotlib.pyplot as plt
import caffe
import os
import scipy.io
import shutil
from class_100_single_plot_roc import drawroc
# Make sure that caffe is on the python path:
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tsne.tsne_1 import tsnepng
caffe.set_mode_gpu()
#caffe.set_mode_cpu()
caffe.set_device(0)

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
    params = {'legend.fontsize':20}
    plt.rcParams.update(params)
    # plt.rcParams['image.cmap'] = 'gray'
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
    mat = []
    # lines =  labelfile(labelfilepath)
    #  print lines
    labels = []
    pred = []
    predroc= []
    nn = 0
    caffe.set_mode_gpu()
    caffe.set_device(0)
    allimages= GetFileList(inputimagedir, [])
    print allimages
    testimages =allimages
    # from random import shuffle
    import random
    # print allimages
    random.shuffle(testimages)
    errorimagelist="./error/"+outdir.split(".")[0]
    if not os.path.exists(errorimagelist):
        os.makedirs(errorimagelist)

    # print testimages

    for image in testimages:
        print image
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
           print out['prob'].argmax(), gtlabel
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
            with open(outdir,'w') as f:
                scipy.io.savemat(f, {'data' :mat,'labels':labels}) #append
        nn += 1
    # print pred.shape
    # tsnepng(mat,labels,"gootsne_"+outdir)
    print "tsnepng=========================>>>>"
    print len(labels), len(predroc)
    drawroc(labels,predroc,"zoomroc_"+outdir.split('.')[0]+".png")
    print "roc=========================>>>>"
    print (classification_report(labels,pred))
    text_file = open(outdir.split('.')[0]+".txt", "w")
    text_file.write(outdir.split('.')[0]+" model\n")
    text_file.write(classification_report(labels,pred))
    import pickle
    with open("pred.pkl","wb") as f:
        pickle.dump(mat,f)
    with open("true","wb") as f:
        pickle.dump(labels,f)
    with open(outdir,'w') as f:
        scipy.io.savemat(f, {'data' :mat,'labels':labels}) #append
    cm=confusion_matrix(pred, labels)
    with open(outdir.split(".")[0]+".pkl","wb") as f:
        pickle.dump(cm,f)
        print cm
        np.savetxt(outdir.split(".")[0]+"mfse"+".csv", cm, delimiter=",")
    #inputimagedir = sys.argv[1]
    # tsnepng(mat,labels,"gootsne_"+outdir)
    # print "tsnepng=========================>>>>"
    # drawroc(labels,pred,"zoom2roc_"+outdir.split('.')[0]+".png")
    # print "roc=========================>>>>"
    # import pickle
    # with open("pred.pkl","wb") as f:
        # pickle.dump(mat,f)
    # with open("true","wb") as f:
        # pickle.dump(labels,f)
    # with open(outdir,'w') as f:
        # scipy.io.savemat(f, {'data' :mat,'labels':labels}) #append
    # cm=confusion_matrix(pred, labels)
    # with open(outdir.split(".")[0]+".pkl","wb") as f:
        # pickle.dump(cm,f)
        # print cm
def predict_image_top_k(net, transformer, inputimagedir,outdir,labelfilepath,layername, k = 5):
    mat = []
    # lines =  labelfile(labelfilepath)
    #  print lines
    labels = []
    pred = []
    predroc= []
    pred_top_k= []
    nn = 0
    caffe.set_mode_gpu()
    caffe.set_device(0)
    allimages= GetFileList(inputimagedir, [])
    # print allimages
    testimages =allimages
    # from random import shuffle
    import random
    # print allimages
    random.shuffle(testimages)
    errorimagelist="./error/"+outdir.split(".")[0]
    if not os.path.exists(errorimagelist):
        os.makedirs(errorimagelist)

    # print testimages
    errornumber = 0

    for image in testimages:
        # print image
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
           errornumber += 1
           # print out['prob'].argmax(), gtlabel
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
        res = net.blobs[net.blobs.keys()[-1]].data[0].flatten()
        pred_top_k.append(res.argsort()[-k:][::-1])
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
            with open(outdir,'w') as f:
                scipy.io.savemat(f, {'data' :mat,'labels':labels}) #append
        nn += 1
    top_k_true = 0
    for l, p_k in zip(labels, pred_top_k):
        if  l in p_k:
            top_k_true += 1
    print "----------------------top k accuracy--------------------------",top_k_true/np.float(len(labels))
    # print "----------------------top k accuracy--------------------------",top_k_true/np.float(len(labels))
    print outdir
    np.savetxt(outdir.split(".")[0]+"top_k"+".txt", [top_k_true/np.float(len(labels)), (np.float(len(labels))-errornumber)/np.float(len(labels)) ] )

    # print pred.shape
    # tsnepng(mat,labels,"gootsne_"+outdir)
    print "tsnepng=========================>>>>"
    # print len(labels), len(predroc)
    drawroc(labels,predroc,"zoomroc_"+outdir.split('.')[0]+".png")
    print "roc=========================>>>>"
    print (classification_report(labels,pred))
    text_file = open(outdir.split('.')[0]+".txt", "w")
    text_file.write(outdir.split('.')[0]+" model\n")
    text_file.write(classification_report(labels,pred))
    import pickle
    with open("pred.pkl","wb") as f:
        pickle.dump(mat,f)
    with open("true","wb") as f:
        pickle.dump(labels,f)
    with open(outdir,'w') as f:
        scipy.io.savemat(f, {'data' :mat,'labels':labels}) #append
    cm=confusion_matrix(pred, labels)
    with open(outdir.split(".")[0]+".pkl","wb") as f:
        pickle.dump(cm,f)
        print cm
        np.savetxt(outdir.split(".")[0]+"mfse"+".csv", cm, delimiter=",")
    #inputimagedir = sys.argv[1]
    # tsnepng(mat,labels,"gootsne_"+outdir)
    # print "tsnepng=========================>>>>"
    # drawroc(labels,pred,"zoom2roc_"+outdir.split('.')[0]+".png")
    # print "roc=========================>>>>"
    # import pickle
    # with open("pred.pkl","wb") as f:
        # pickle.dump(mat,f)
    # with open("true","wb") as f:
        # pickle.dump(labels,f)
    # with open(outdir,'w') as f:
        # scipy.io.savemat(f, {'data' :mat,'labels':labels}) #append
    # cm=confusion_matrix(pred, labels)
    # with open(outdir.split(".")[0]+".pkl","wb") as f:
        # pickle.dump(cm,f)
        # print cm
def batch_extrac_featuretomat():
    #alexnet
    alexnetpath=r"./dvns/imagenet/alexnet/"
    googlenetpath=r"./dvns/imagenet/googlenet/"
    residualnetpath=r"./dvns/imagenet/residualnet/"
    vggnetpath=r"./dvns/imagenet/vggnet/"
    ninnetpath=r"./dvns/imagenet/ninnet/"
    caffenetpath=r"./dvns/imagenet/caffenet/"
    simplenetpath=r"./dvns/imagenet/simplenet/"
    lenetnetpath=r"./dvns/imagenet/lenet/"
    highwaynetpath=r"./dvns/imagenet/highway/"
    #cifar10netpath=r"./dvns/cifar10/cifar_cifar10/"
    svhn_cifar10netpath=r"./dvns/svhn/cifar10net/"
    svhn_googlenetpath=r"./dvns/svhn/googlenet/"
    svhn_alexnetpath=r"./dvns/svhn/alexnet/"


              # googlenetpath+"bvlc_googlenet_iter_520000.caffemodel",\
              # googlenetpath+"oribvlc_googlenet_iter_520000.caffemodel",\
    # modelist=[alexnetpath+"oriciafr10caffe_alexnet_train_iter_390000.caffemodel",\
              # alexnetpath+"dvnciafr10caffe_alexnet_train_iter_450000.caffemodel",\
    modelist=[alexnetpath+"imagenet_caffe_alexnet_train_iter_80000.caffemodel",\
              alexnetpath+"alex_smalldataset_dvn_iter_294000.caffemodel",\
              # googlenetpath+"bvlc_googlenet_data_generate_iter_5120000.caffemodel",\
              # googlenetpath+"bvlc_googlenet_data_ori_iter_5360000.caffemodel",\
              #googlenetpath+"119_bvlc_googlenet_iter_225681.caffemodel",\
              googlenetpath+"124gen_bvlc_googlenet_iter_1240000.caffemodel",\
              googlenetpath+"119_bvlc_googlenet_iter_400000.caffemodel",\
              #residualnetpath+"resnet_50_iter_150000.caffemodel",\
              residualnetpath+"resnet_50_iter_700000.caffemodel",\
              residualnetpath+"1212resnet_50_gen_dvn_iter_150000.caffemodel",\
              #residualnetpath+"dvnresnet_50_iter_150000.caffemodel",\
              vggnetpath+"vgg16_small_dateset_iter_40000.caffemodel",\
              vggnetpath+"vgg16_small_dateset_iter_40000.caffemodel",\
              ninnetpath+"nin_small_dataset_iter_450000.caffemodel",\
              ninnetpath+"nin_small_dataset_iter_450000.caffemodel",\
              caffenetpath+"gen_caffenet_iter_150000.caffemodel",\
              caffenetpath+"ori_caffenet_iter_170000.caffemodel",\
              simplenetpath+"gen_simplenet_iter_100000.caffemodel",\
              simplenetpath+"ori_simplenet_iter_100000.caffemodel",\
              lenetnetpath+"gen_lenet_iter_50000.caffemodel",\
              lenetnetpath+"ori_lenet_iter_50000.caffemodel",\
              highwaynetpath+"ori_highway_iter_76225.caffemodel",\
              highwaynetpath+"gen_highway_iter_75458.caffemodel",\
              #vggnetpath+"cifar10_full_iter_60000.caffemodel",\
              svhn_cifar10netpath+"svhn_combine_47_96_easy_iter_300000.caffemodel",\
              svhn_cifar10netpath+"svhn_ori_iter_60000.caffemodel.h5",\
              # svhn_alexnetpath+"caffe_alexnet_train_ori_iter_450000.caffemodel",\
              svhn_alexnetpath+"svhn_caffe_alexnet_train_iter_120000.caffemodel",\
              svhn_alexnetpath+"svhn_caffe_alexnet_train_iter_120000.caffemodel",\
              svhn_googlenetpath+"svhn_bvlc_googlenet_ge_iter_426467.caffemodel",\
              svhn_googlenetpath+"svhn_bvlc_googlenet_iter_110000.caffemodel"\

              ]
    datalist=["./imagenetdata/val/",
              "./imagenetdata/val/",
              "./imagenetdata/val/",
              "./imagenetdata/val/",
              "./imagenetdata/val/",
              "./imagenetdata/val/",
              "./imagenetdata/val/",
              "./imagenetdata/val/",
              "./imagenetdata/val/",
              "./imagenetdata/val/",
              "./imagenetdata/val/",
              "./imagenetdata/val/",
              "./imagenetdata/val/",
              "./imagenetdata/val/",
              "./imagenetdata/val/",
              "./imagenetdata/val/",
              "./generatemat/data/svhn_ori_val/",
              "./generatemat/data/svhn_ori_val/",
              ]
    deploylist=[alexnetpath+"deploy.prototxt",
                alexnetpath+"deploy.prototxt",
                googlenetpath+"deploy.prototxt",
                googlenetpath+"deploy.prototxt",
                residualnetpath+"deploy.prototxt",
                residualnetpath+"deploy.prototxt",
                vggnetpath+"deploy.prototxt",
                vggnetpath+"deploy.prototxt",
                ninnetpath+"deploy.prototxt",
                ninnetpath+"deploy.prototxt",
                caffenetpath+"deploy.prototxt",
                caffenetpath+"deploy.prototxt",
                simplenetpath+"deploy.prototxt",
                simplenetpath+"deploy.prototxt",
                lenetnetpath+"deploy.prototxt",
                lenetnetpath+"deploy.prototxt",
                highwaynetpath+"deploy.prototxt",
                highwaynetpath+"deploy.prototxt",
                svhn_alexnetpath+"deploy.prototxt",
                svhn_alexnetpath+"deploy.prototxt",
                svhn_googlenetpath+"deploy.prototxt",
                svhn_googlenetpath+"deploy.prototxt"
                ]
    #meanlist=[alexnetpath+"patchcifa10_256_mean.binaryproto",
    meanlist=[alexnetpath+"patchsmall_dataset_mean_100.binaryproto",
              alexnetpath+"patchsmall_dataset_mean_100.binaryproto",
              alexnetpath+"patchsmall_dataset_mean_100.binaryproto",
              alexnetpath+"patchsmall_dataset_mean_100.binaryproto",
              alexnetpath+"patchsmall_dataset_mean_100.binaryproto",
              alexnetpath+"patchsmall_dataset_mean_100.binaryproto",
              alexnetpath+"patchsmall_dataset_mean_100.binaryproto",
              alexnetpath+"patchsmall_dataset_mean_100.binaryproto",
              alexnetpath+"patchsmall_dataset_mean_100.binaryproto",
              alexnetpath+"patchsmall_dataset_mean_100.binaryproto",
              alexnetpath+"patchsmall_dataset_mean_100.binaryproto",
              alexnetpath+"patchsmall_dataset_mean_100.binaryproto",
              alexnetpath+"patchsmall_dataset_mean_100.binaryproto",
              alexnetpath+"patchsmall_dataset_mean_100.binaryproto",
              alexnetpath+"patchsmall_dataset_mean_100.binaryproto",
              alexnetpath+"patchsmall_dataset_mean_100.binaryproto",
              alexnetpath+"patchsmall_dataset_mean_100.binaryproto",
              alexnetpath+"patchsmall_dataset_mean_100.binaryproto",
              alexnetpath+"patchsmall_dataset_mean_100.binaryproto",
              svhn_cifar10netpath+"patchcombine_47_96_easy_256_mean.binaryproto",
              svhn_cifar10netpath+"patchcombine_47_96_easy_256_mean.binaryproto",
              svhn_cifar10netpath+"patchcombine_47_96_easy_256_mean.binaryproto",
              svhn_cifar10netpath+"patchcombine_47_96_easy_256_mean.binaryproto",
              svhn_cifar10netpath+"patchcombine_47_96_easy_256_mean.binaryproto"
              ]
    shapelists=[[10,3,227,227],[10,3,227,227],[32,3,224,224],[32,3,224,224],
		[1,3,224,224],[1,3,224,224],
                [1,3,224,224],[1,3,224,224],[1, 3, 224, 224],[1, 3, 224, 224],
                [1,3,227,227],[1,3,227,227],
               [1,3,256,256],[1,3,256,256],
                [1,3,256,256],[1,3,256,256],
                [10,3,224,224],[10,3,224,224]]
    labellist=["cifa10_valdst.csv",
               "cifa10_valdst.csv",
               "cifa10_valdst.csv",
               "cifa10_valdst.csv",
               "cifa10_valdst.csv",
               "cifa10_valdst.csv",
               "cifa10_valdst.csv",
               "cifa10_valdst.csv",
               "cifa10_valdst.csv",
               "cifa10_valdst.csv",
               "cifa10_valdst.csv",
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
    outlist=["imagenet_alex_dvn.mat","imagenet_alex_ori.mat","imagenet_googlenet_dvn.mat",
             "imagenent_googlenet_ori.mat","imagenet_residualnet_ori.mat","imagenet_residualnet_15k_dvn.mat","imagenet_vggnet_dvn.mat","imagenet_vggnet_dvn.mat", "imagenet_nin_ori.mat",
             "imagenet_nin_dvn.mat","imagenet_caffenetdvn.mat","imagenet_caffenetori.mat","imagenet_simplenetori.mat","imagenet_simplenetdvn.mat","imagenet_lenetdvn.mat",
             "imagenet_lenetori.mat", "imagenet_highwayori.mat", "imagenet_highwaydvn.mat",
             "svhn_google_dvn.mat","svhn_google_ori.mat"
             ]
    layernamelist=["fc8","fc8","loss3/classifier","loss3/classifier","fc1000","fc1000","fc8new","fc8new",
                   "pool4","pool4","fc8","fc8","loss3/classifier","loss3/classifier",
                   "pool4","pool4","fc8","fc8","loss3/classifier","loss3/classifier",
                   "pool4","pool4","fc8","fc8","loss3/classifier","loss3/classifier"]

    import traceback
    # for i in range(0,len(modelist)):
    #for i in range(3, 4):
    for i in range(0, 2):
    #for i in range(1,6):
    # for i in range(0,1):
        # if i<4 and i>1:
            # continue
    # for i in range(2,4):
        try:
          print modelist[i]
          net,transformer=loadmodel(modelpath=modelist[i],deployroot=deploylist[i],
                                    meanroot=meanlist[i],shapelist=shapelists[i])
          # image2mat(net,transformer,datalist[i],outlist[i],labellist[i],layernamelist[i])
          predict_image_top_k(net,transformer,datalist[i],outlist[i],labellist[i],layernamelist[i])
        except Exception as e:
          print e
          print traceback.format_exc()
          return
          # break
          continue
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
