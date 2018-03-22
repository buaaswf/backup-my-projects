#!/usr/bin/env python
# encoding: utf-8

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from numpy.linalg import norm
import numpy as np
plt.switch_backend("Agg")
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
# inputdir = "./csvs/cifar10oversamplinganddvn/"
# inputdir = "./csvs/mnist/ministallnets/"
# inputdir = "./csvs/cifar10andsvhn/"
# inputdir = "./csvs/cifar10oversamplinganddvn/"
inputdir = "./csvs/torch/"
csvlist = GetFileList(inputdir,[])
for csvfile in csvlist:
    print csvfile
    array = pd.read_csv(csvfile, sep=',',header=None)
    array = array.values
    # array = cv2.blur(array, (10, 10))
    name = csvfile.split("/")[-1].split(".")[0]
    print array
# print array
# print array.shape
# df_cm = pd.DataFrame(array, index = [i for i in "'ABCDEFGHIJ'"],
                    # columns = [i for i in "'ABCDEFGHIJ'"])
    # df_cm = pd.DataFrame(array, index =[i for i in "01"] ,columns = [i for i in "01"])
    df_cm = pd.DataFrame(array)
    df_cm = df_cm.values
    df_cm = df_cm.astype(float)
    df_cm_rowsums = df_cm.sum(axis=1)
    df_cm = df_cm/df_cm_rowsums[:,np.newaxis]
    np.savetxt("csvs\\torch\\res"+name+".csv", df_cm, delimiter=",")
    # linfnorm = norm(df_cm, axis=1, ord=np.inf)
    # print linfnorm
    # df_cm.astype(np.float) / linfnorm
    plt.figure(figsize = (10,7))
    print df_cm
    # sn.heatmap(df_cm, cmap="jet",annot=True)
    sn.heatmap(df_cm, annot=True,fmt=".3g")
    plt.savefig(os.path.join(inputdir,name+".png"),bbox_inches='tight', pad_inches = 0 )
