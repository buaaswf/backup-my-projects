import numpy as np
import sys
import scipy.io
#import matplotlib.pyplot as plt

sys.path.insert(0,"/home/user/caffe/python")
# Make sure that caffe is on the python path:
# caffe_root = '../../../'  # this file is expected to be in {caffe_root}/examples
import sys
import os
# sys.path.insert(0, "/home/s.li/2017/gpu4/caffe/" + 'python')
import matplotlib.pyplot as plt

import caffe
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
'''
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(iris.target_names))
plt.xticks(tick_marks, iris.target_names, rotation=45)
plt.yticks(tick_marks, iris.target_names)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig("confusionmatrx.jpg")
'''

def load_models(modelpath = "models/idface/",deploypath = "models/",meanfacepath =
                "data/idface/",shapelist=[63,3,55,55]):

    caffe_root = ''  # this file is expected to be in {caffe_root}/examples
    sys.path.insert(0, caffe_root + 'python')
    # plt.rcParams['figure.figsize'] = (10, 10)
    # plt.rcParams['image.interpolation'] = 'nearest'
    # plt.rcParams['image.cmap'] = 'gray'
    model = modelpath # caffemodelroot
    if not os.path.isfile(caffe_root + model):
        print("Downloading pre-trained CaffeNet model...")
    caffe.set_mode_cpu()
    net = caffe.Net( deploypath, modelpath, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( caffe_root+meanfacepath , 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    out = arr[0]
    if (shapelist[2]!=28):
        transformer.set_mean('data', out.mean(1).mean(1))   # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    #  transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
    net.blobs['data'].reshape(shapelist[0],shapelist[1],shapelist[2],shapelist[3])
    return net, transformer


def generateprediction(imglist,net,transformer,shapelist):
    prediction =[]
    onehotpred=[]
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    IMAGE_FILE = imglist


    caffe.set_mode_gpu()
    caffe.set_device(0)
    for img in imglist:
        #print img
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img,color=True))
    #    net.blobs['data'].reshape(shapelist[0],shapelist[1],shapelist[2],shapelist[3])
    out = net.forward()
    #  print out['prob']
    prediction.append( np.argmax(out['prob']))  # predict takes any number of images, and formats them for the Caffe net automatically
    onehotpred.append(out['prob'])  # predict takes any number of images, and formats them for the Caffe net automatically
    return prediction,onehotpred
def generatefeature(imglist,net,transformer,labels,shapelist,layername,outmat):
    prediction =[]
    pred=[]
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    caffe.set_mode_gpu()
    caffe.set_device(0)
    IMAGE_FILE = imglist
    count=0
    for img in imglist:
        print img
        count+=1
        #print "feature",img,
        #count,len(imglist)
        #print shapelist
        if (shapelist[1]==3):
            net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img,color=True))
        else:
            net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img,color=False))
        #net.blobs['data'].reshape(shapelist[0],shapelist[1],shapelist[2],shapelist[3])
        caffe.set_device(0)
        out = net.forward()
        pred.append(str(out['prob'].argmax()))
        #  feature = net.blobs['fc7'].data[0]
        feature = net.blobs[layername].data[0]
        featline = feature.flatten()
        #mat.append(featline)
        #print out['prob']
        prediction.append(featline)  # predict takes any number of images, and formats them for the Caffe net automatically
        nn = 0
        #if (nn%100==0):
    with open(outmat,'w') as f:
        scipy.io.savemat(f, {'data' :prediction,'labels':labels})
    return pred,prediction
def generatefeaturenopredcition(imglist,net,transformer,labels,shapelist,layername,outmat):
    prediction =[]
    pred=[]
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    caffe.set_mode_gpu()
    caffe.set_device(0)
    IMAGE_FILE = imglist
    count=0
    #for img in imglist[:100]+imglist[-100:]:
    for img in imglist:
        print img
        count+=1
        #print "feature",img,
        #count,len(imglist)
        #print shapelist
        if (shapelist[1]==3):
            net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img,color=True))
        else:
            net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img,color=False))
        #net.blobs['data'].reshape(shapelist[0],shapelist[1],shapelist[2],shapelist[3])
        caffe.set_device(0)
        out = net.forward()
        pred.append(str(out['ip1']))
        #  feature = net.blobs['fc7'].data[0]
        feature = net.blobs[layername].data[0]
        featline = feature.flatten()
        #mat.append(featline)
        #print out['prob']
        prediction.append(featline)  # predict takes any number of images, and formats them for the Caffe net automatically
        nn = 0
    with open(outmat,'w') as f:
        scipy.io.savemat(f, {'data' :prediction,'labels':labels})
    return prediction
def generatefeature_3channels(imglist,net,transformer,labels, mean="python/caffe/imagenet/ilsvrc_2012_mean.npy"):
    prediction =[]
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    IMAGE_FILE = imglist
    net.set_mode_gpu()
    net.set_device(0)
    for img in imglist:
        #net.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))
        # load the mean ImageNet image (as distributed with Caffe) for subtraction
        mu = np.load(caffe_root + mean)
        mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img,color=False))
        net.blobs['data'].reshape(64,1,28,28)
        out = net.forward()
        feature = net.blobs['ip2'].data[0]
        featline = feature.flatten()
        #mat.append(featline)
        #print out['prob']
        prediction.append(featline)  # predict takes any number of images, and formats them for the Caffe net automatically
        nn = 0
        #if (nn%100==0):
    with open('google_dvn_mnist.mat','w') as f:
        print (data)
        scipy.io.savemat(f, {'data' :prediction,'labels':labels})
    #return prediction

def GetFileList(dir,fileList):
    new_Dir = dir
    if os.path.isfile(dir):
        fileList.append(dir.decode('gbk'))
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            if s.endswith(".txt") or s.endswith(".sh") or s.endswith(".py"):
                continue
            #if int(s)>998 and int(s) < 1000:
            newDir=os.path.join(dir,s)
            GetFileList(newDir, fileList)
    return fileList
import re
def mklabel(fileList,dst):
    label = []
    # pattern = re.compile(u'/'+dst+"/(.*?)/")
    # pattern = re.compile(u'\'+dst+"/(.*?)/")
    # pattern = re.compile(u'\'+dst+"\(.*?)\")
    for file in fileList:
        # print file
        label.append(file.split("/")[-2])
        # label.append(pattern.findall(file.strip())[0])
    #print label
    label = map(int, label)
    return label
def treeimages2confusionmatrix(imagepath="images"):
    from sklearn.metrics import confusion_matrix
    y_true = []
    y_pred=[]
    #  shapelist=[10,3,227,227]
    shapelist=[10,3,224,224]
    #net,transformer=load_models(modelpath = "./mninstmodel/lenet_iter_10000.caffemodel",deploypath = "./mnistmodel/lenet_deploy.prototxt",meanfacepath ="",shapelist=[64,1,28,28])
    #  net,transformer=load_models(modelpath = "/home/g206/work_sdf/caffe1225/caffe/analysisfeatures/dvns/mnist/dvnmnistcaffe_alexnet_train_iter_420000.caffemodel",deploypath = "/home/g206/work_sdf/caffe1225/caffe/analysisfeatures/dvns/mnist/alex_deploy.prototxt",meanfacepath ="",shapelist=shapelist)
    #  net,transformer=load_models(modelpath = "/home/g206/work_sdf/caffe1225/caffe/analysisfeatures/dvns/\
    #  mnist/orimnistcaffe_alexnet_train_iter_400000.caffemodel",deploypath = "/home/g206/work_sdf/caffe1225/caffe/analysisfeatures/dvns\
    #  /mnist/alex_deploy.prototxt",meanfacepath ="",shapelist=shapelist)
    #  net,transformer=load_models(modelpath = "/home/g206/work_sdf/caffe1225/caffe/analysisfeatures/dvns/mnist/orimnitbvlc_googlenet_iter_400000.caffemodel",deploypath = "/home/g206/work_sdf/caffe1225/caffe/analysisfeatures/dvns/mnist/google_deploy.prototxt",meanfacepath ="",shapelist=shapelist)
    net,transformer=load_models(modelpath = "/home/g206/work_sdf/caffe1225/caffe/analysisfeatures/dvns/mnist/mnistgenerate_bvlc_googlenet_iter_480000.caffemodel",deploypath = "/home/g206/work_sdf/caffe1225/caffe/analysisfeatures/dvns/mnist/google_deploy.prototxt",meanfacepath ="",shapelist=shapelist)
    #  net,transformer=load_models(modelpath = "./mnistmodel/lenet_dvn__iter_500000.caffemodel",deploypath = "./mnistmodel/lenet_deploy.prototxt",meanfacepath ="",shapelist=[64,1,28,28])
    # net,transformer=load_models(modelpath = "./mnistmodel/lenet_iter_10000.caffemodel",deploypath = "./mnistmodel/lenet_deploy.prototxt",meanfacepath ="",shapelist=[64,1,28,28])
    imglist =GetFileList(imagepath,[])
    y_true=mklabel(imglist,"mnist_test")
    #  y_pred,y_onehot=generateprediction(imglist,net,transformer,shapelist)
    generatefeature(imglist,net,transformer,y_true,shapelist)
    import pickle
    with open("googlepredori.pkl", 'wb') as f:
        pickle.dump(y_pred, f)
    with open("googletrueori.pkl", 'wb') as f:
        pickle.dump(y_true,f)
    with open("googleonehotori.pkl", 'wb') as f:
        pickle.dump(y_onehot,f)
    cm=confusion_matrix(y_pred, y_true)
    with open("googleconfusionmatrx.pkl","wb") as f:
        print cm
        pickle.dump( confusion_matrix(y_pred, y_true),f)
def batch_treeimages2confusionmatrix(imagepath="images"):
    #netpath="/home/g206/work_sdf/caffe1225/caffe/analysisfeatures/dvns/mnist"
    netpath="./dvns/mnist/"
    modelist=["dvnmnistcaffe_alexnet_train_iter_420000.caffemodel","orimnistcaffe_alexnet_train_iter_400000.caffemodel",
              "lenet_dvn__iter_500000.caffemodel","lenet_oridata__iter_500000.caffemodel",
              "orimnitbvlc_googlenet_iter_400000.caffemodel","mnistgenerate_bvlc_googlenet_iter_480000.caffemodel",
            ]
    deploylist=["alex_deploy.prototxt","alex_deploy.prototxt","lenet.prototxt","lenet.prototxt","google_deploy.prototxt","google_deploy.prototxt",]
    shapelists=[[10,3,227,227],[10,3,227,227],[64,1,28,28],[64,1,28,28],[10,3,224,224],[10,3,224,224]]
    netlayerlist=["fc8","fc8","ip2","ip2","loss3/classifier","loss3/classifier"]
    outnamelist=["mnistlenetdvn.mat","mnistlenetori.mat","mnistalexdvn.mat","mnistalexori.mat","mnistgoogledvn.mat","mnistgoogleori.mat"]
    meanlist=["patchmnistimages_256_mean.binaryproto","patchmnistimages_256_mean.binaryproto","patchmnistimages_256_mean.binaryproto"
              ,"patchmnistimages_256_mean.binaryproto","patchmnistimages_256_mean.binaryproto","patchmnistimages_256_mean.binaryproto"]

    from sklearn.metrics import confusion_matrix
    y_true = []
    #networklist=["lenet","alext","google"]
    #net,transformer=load_models(modelpath = "./mninstmodel/lenet_iter_10000.caffemodel",deploypath = "./mnistmodel/lenet_deploy.prototxt",meanfacepath ="",shapelist=[64,1,28,28])
    #  net,transformer=load_models(modelpath = "/home/g206/work_sdf/caffe1225/caffe/analysisfeatures/dvns/mnist/orimnistcaffe_alexnet_train_iter_400000.caffemodel",deploypath = "/home/g206/work_sdf/caffe1225/caffe/analysisfeatures/dvns/mnist/alex_deploy.prototxt",meanfacepath ="",shapelist=shapelist)
    #orimnitbvlc_googlenet_iter_400000.caffemodel
    #  net,transformer=load_models(modelpath = "/home/g206/work_sdf/caffe1225/caffe/analysisfeatures/dvns
    imglist =GetFileList(imagepath,[])
    i=0
    #  /mnist/mnistgenerate_bvlc_googlenet_iter_480000.caffemodel",deploypath = "/home/g206/work_sdf/caffe1225/caffe/analysisfeatures/dvns/mnist/alex_deploy.prototxt",meanfacepath ="",shapelist=shapelist)
    from tsne.tsne_1 import tsnepng
    from mnist_single_plot_roc import drawroc
    import traceback
    for i in range(0,len(modelist)):
    # for i in range(0, 2):
        try:
            pred=[]
            prediction=[]
            y_true=[]
            net,transformer=load_models(modelpath =netpath+ modelist[i],
                                        deploypath = netpath+deploylist[i],meanfacepath=netpath+meanlist[i],shapelist=shapelists[i])
            y_true=mklabel(imglist,"mnist_test")
            print "generating feature for model {0}....".format(modelist[i])
            pred,prediction=generatefeature(imglist,net,transformer,y_true,shapelists[i],netlayerlist[i],outnamelist[i])
            print len(y_true), len(pred)
            print "roc"+outnamelist[i].split(".mat")[0]+".png"
            drawroc(y_true,prediction,"roc"+outnamelist[i].split(".mat")[0]+".png")
            #tsnepng(prediction,y_true,"tsne_"+outnamelist[i].split(".mat")[0]+".png")
            import pickle
            with open(outnamelist[i].split(".")[0]+"pred.pkl", 'wb') as f:
                pickle.dump(pred, f)
            with open(outnamelist[i].split(".")[0]+"true.pkl", 'wb') as f:
                pickle.dump(y_true,f)
            y_true=np.asarray(y_true,dtype=int)
            pred=np.asarray(pred,dtype=int)
            print pred
            print y_true
            cm=confusion_matrix(pred, y_true)
            print cm
            with open(outnamelist[i].split(".")[0]+ "_cm.pkl", "wb") as f:
                pickle.dump(cm, f)
                print cm
                np.savetxt(outnamelist[i].split(".")[0] + "_cm" + ".csv", cm, delimiter=",")
        except Exception as e:
            print e
            traceback.print_exc()
            continue
def image2features(imagepath,netpath):
    #netpath="./dvns/mnistanisg_result/"
    modelist=["mnistanisg_iter_10000.caffemodel","mnistorig_iter_10000.caffemodel",
              "lenet_dvn__iter_500000.caffemodel","lenet_oridata__iter_500000.caffemodel",
              "orimnitbvlc_googlenet_iter_400000.caffemodel","mnistgenerate_bvlc_googlenet_iter_480000.caffemodel",
            ]
    deploylist=["mnist_deploy.prototxt","mnist_deploy.prototxt","lenet.prototxt","lenet.prototxt","google_deploy.prototxt","google_deploy.prototxt",]
    shapelists=[[1,3,256,256],[1,3,256,256],[64,1,28,28],[64,1,28,28],[10,3,224,224],[10,3,224,224]]
    netlayerlist=["ip1","ip1","ip2","ip2","loss3/classifier","loss3/classifier"]
    outnamelist=["mnistcenter_loss_dvn.mat","mnistcenter_loss_ori.mat","mnistalexdvn.mat","mnistalexori.mat","mnistgoogledvn.mat","mnistgoogleori.mat"]
    meanlist=["mnista_mean.binaryproto","mnista_mean.binaryproto","patchmnistimages_256_mean.binaryproto"
              ,"patchmnistimages_256_mean.binaryproto","patchmnistimages_256_mean.binaryproto","patchmnistimages_256_mean.binaryproto"]

    from sklearn.metrics import confusion_matrix
    y_true = []
    imglist =GetFileList(imagepath,[])
    i=0
    from tsne.tsne_1 import tsnepng
    from mnist_single_plot_roc import drawroc
    import traceback
    #for i in range(0,len(modelist)):
    for i in range(1,2):
        try:
            pred=[]
            prediction=[]
            y_true=[]
	    colors = np.arange(10)
	    print colors
	    p_color=[]

            net,transformer=load_models(modelpath =netpath+ modelist[i],
                                        deploypath = netpath+deploylist[i],meanfacepath=netpath+meanlist[i],shapelist=shapelists[i])
            y_true=mklabel(imglist,"mnist_test")
#	    y_true=y_true[:100]+y_true[-100:]
	    for c in y_true:
		p_color.append(colors[int(c)])
            print "generating feature for model {0}....".format(modelist[i])
            prediction=generatefeaturenopredcition(imglist,net,transformer,y_true,shapelists[i],netlayerlist[i],outnamelist[i])
            # print len(y_true), len(pred)
            # print "roc"+outnamelist[i].split(".mat")[0]+".png"
            # drawroc(y_true,prediction,"roc"+outnamelist[i].split(".mat")[0]+".png")
            #tsnepng(prediction,y_true,"tsne_"+outnamelist[i].split(".mat")[0]+".png")
            import pickle
            with open(outnamelist[i].split(".")[0]+"pred.pkl", 'wb') as f:
                pickle.dump(prediction, f)
            with open(outnamelist[i].split(".")[0]+"y_true.pkl", 'wb') as f:
                pickle.dump(y_true, f)
            # t = np.arange(10)
            #for i ,co in zip(prediction, y_true):
	#	print co
        #        plt.scatter(i[0], i[1], s=1,c=colors[int(co)],alpha=0.5)
        #        plt.savefig("res1212.png")
        #    for i ,co in zip(prediction, y_true):
	#	print co
	    print colors
	    print p_color
	    prediction =np.transpose( np.asarray(prediction))
            plt.scatter(prediction[0], prediction[1],s=1,c=np.asarray(p_color),alpha=0.5)
            plt.savefig("res1212ori.png")
            # with open(outnamelist[i].split(".")[0]+"true.pkl", 'wb') as f:
            #     pickle.dump(y_true,f)
            # y_true=np.asarray(y_true,dtype=int)
            # pred=np.asarray(y_true,dtype=int)
            # print pred
            # print y_true
            # cm=confusion_matrix(pred, y_true)
            # print cm
            # with open(outnamelist[i].split(".")[0]+ "_cm.pkl", "wb") as f:
            #     pickle.dump(cm, f)
            #     print cm
            #     np.savetxt(outnamelist[i].split(".")[0] + "_cm" + ".csv", cm, delimiter=",")
        except Exception as e:
            print e
            traceback.print_exc()
            continue
if __name__=="__main__":
    # batch_treeimages2confusionmatrix(".\\mnist_test\\")
    #image2features("./mnist_test/","dvns/mnistanisg_result/")
    image2features("./mnist_test/","dvns/mnistorig_result/")
