#!/usr/bin/env python
# encoding: utf-8
import os
import cv2
import numpy as np
import  scipy.io
from PIL import Image
from skimage import data, img_as_float

from skimage.restoration import denoise_tv_chambolle, denoise_bilateral

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
def checktwodirs(originimagepath, segmentationimagepath):
    #  originimage = GetFileList(originimage, []])
    segmentationimage = GetFileList(segmentationimagepath, [])
    list1 = []
    list2 = []
    for seg in segmentationimage:
        imageid = seg.split("/")[-1].split(".")
        originimageid  = imageid[0] + ".jpg"
        import os.path
        oriimage = originimagepath+originimageid
        print oriimage
        if (os.path.isfile(oriimage)):
            print oriimage
            list1.append(oriimage)
            list2.append(seg)
    import csv
    with open('text.csv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(list1,list2))
def resizeimage(srcpath, dstpath):
    imagelist = GetFileList(srcpath,[])
    height, width = 360, 480
    for img in imagelist:
        #  print img
        image = cv2.imread(img,cv2.IMREAD_UNCHANGED)
        newimage = cv2.resize(image,(height, width),interpolation = cv2.INTER_CUBIC)
        newimage = np.rot90(newimage,3)
        cv2.imwrite(dstpath+img.split("/")[-1], newimage)
def normalise(srcpath, dstpath):
    imagelist = GetFileList(srcpath,[])
    height, width = 360, 480
    for img in imagelist:
        #  print img
        image = cv2.imread(img,0)
        #  newimage = cv2.resize(image,(height, width),interpolation = cv2.INTER_CUBIC)
        #  newimage = np.rot90(newimage,3)
        cv2.imwrite(dstpath+img.split("/")[-1], image)
def grayimage(srcpath, dstpath):
    imagelist = GetFileList(srcpath,[])
    height, width = 480, 360
    for img in imagelist:
        image = cv2.imread(img)
        gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #  newimage = cv2.resize(image,(height, width),interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(dstpath+img.split("/")[-1], gray)
def rotateimage(srcpath, dstpath):
    imagelist = GetFileList(srcpath,[])
    for img in imagelist:
        image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        image = np.rot90(image,3)
        #  gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #  newimage = cv2.resize(image,(height, width),interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(dstpath+img.split("/")[-1], image)
def grayimagev2(srcpath, dstpath):
    imagelist = GetFileList(srcpath,[])
    height, width = 480, 360
    for img in imagelist:
        image = cv2.imread(img)
        gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #  counts = np.bincount(gray.flatten())
        #  newimage = cv2.resize(image,(height, width),interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(dstpath+img.split("/")[-1], gray)
def colorimage(srcpath, dstpath):
    imagelist = GetFileList(srcpath,[])
    height, width = 480, 360
    for img in imagelist:
        print img
        image = cv2.imread(img,0)
        image = cv2.equalizeHist(image)
        image = denoise_bilateral(image, sigma_range=0.05, sigma_spatial=15,multichannel=False)

        #  image = np.hstack((image,equ)) #stacking images side-by-side
        #  image = cv2.calcHist([image],[0],None,[256],[0,256])
        #  image=cv2.normalize(image,alpha=0, beta=1, ,0,255,cv2.NORM_L2)
        #  print image
        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        #  print image*
        #  gray=cv2.cvtColor(image, cv2.COLOR_2GRAY)
        #  counts = np.bincount(gray.flatten())
        #  newimage = cv2.resize(image,(height, width),interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(dstpath+img.split("/")[-1], image*255)
def jpg2png(srcpath,dstpath):
    imagelist = GetFileList(srcpath,[])
    height, width = 480, 360
    for img in imagelist:
        image = cv2.imread(img)
        #gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #  newimage = cv2.resize(image,(height, width),interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(dstpath+img.split("/")[-1].split(".")+"png", image)
def mat2png(srcpath,dstpath):
    imagelist = GetFileList(srcpath,[])
    height, width = 480, 360
    for img in imagelist:
        image = scipy.io.loadmat(img)['groundTruth'][0][0][0][0][2].astype(np.uint16)
        #  print image.shape
        image = cv2.resize(image,(480, 360),interpolation = cv2.INTER_CUBIC)
        #  print np.max(image)
        #  image = Image.fromarray(image)
        #  image = image.resize((480,360),Image.BILINEAR)
        #  image = np.array(image.getdata(), np.uint8).reshape(image.size[1], image.size[0])
        image = image[np.newaxis,...]
        #  print image
        #  print image.shape
        #gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #  newimage = cv2.resize(image,(height, width),interpolation = cv2.INTER_CUBIC)
        print dstpath+img.split("/")[-1].split(".")[0]
        cv2.imwrite(dstpath+img.split("/")[-1].split(".")[0]+".png", image[0])
        #  print dstpath+img.split("/")[-1]
        #  cv2.imwrite(dstpath+img.split("/")[-1], image[0])
   

def single_filter(image):
    dataimage =  np.asarray(image) 
    dataimage = img_as_float(dataimage)
    newimage = denoise_bilateral(image, sigma_range=0.05, sigma_spatial=5)
    im = PIL.Image.fromarray(newimage)
    return im
def filter_image(srcpath, dstpath):
    imagelist = GetFileList(srcpath,[])
    #height, width = 360, 480
    for img in imagelist:
        #  print img
        image = cv2.imread(img,cv2.IMREAD_UNCHANGED)
	image = img_as_float(image)	
	newimage = denoise_bilateral(image, sigma_range=0.05, sigma_spatial=5)

        #newimage = cv2.resize(image,(height, width),interpolation = cv2.INTER_CUBIC)
        #newimage = np.rot90(newimage,3)
	finaldst = os.path.join(dstpath, img.split("/")[-2])
	if not os.path.exists(finaldst):
	    os.makedirs(finaldst)
        cv2.imwrite(finaldst+"/"+img.split("/")[-1], newimage)

if __name__=="__main__":
    filter_image("/home1/data/imagenet/small_dataset_gen_500/","/home1/data/imagenet/small_dataset_gen_500_anis/")
