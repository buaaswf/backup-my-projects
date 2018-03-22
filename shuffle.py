#!/usr/bin/env python
# encoding: utf-8
import random
import cv2
def shuforder(list1, list2):
    list3 =GetFileList("~/swf_data/nopass/idcard",[])
    l1 = list1[:]
    random.shuffle(l1)
    image1list = []
    image2list= []

    for imagename1 in l1:
        image1list.append(imagename1.split('/')[-1].split('-0'))
    for img in list3:
        image2list.append(img.split('/')[-1].split('-0'))
    for i in range(0,len(list1)-1):
        if image1list[i] in image2list:
            continue
        if  l1[i]==list1[i] and not image1list[i] in image2list:
            l1[i], l1[i+1]= l1[i+1],l1[i]

    print l1[1],list1[1]
    return l1, list2
def GetFileList(dir, fileList):
    import os
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
def generatneworder():
    list1 =GetFileList("same", [])
    list2=GetFileList("cam",[])
    l1,list2 = shuforder(list1,list2)
    for idnew,idold in zip(l1,list1):
        idimage = cv2.imread(idold)
        newidcpath = "diff/"+ idnew.split('/')[-1]
	#print newidcpath
        cv2.imwrite(newidcpath, idimage)
if __name__ =="__main__":
    generatneworder()

