#!/usr/bin/env python
# encoding: utf-8

import os
import scipy.io
import sys
import random
def shufuldata(input,output):
    f = open(input,'r')
    result = f.readlines()
    #  for line in f:
        #  line = f.readline()
        #  result.append(line)
    random.shuffle(result)
    twfile = open(output+"train.txt",'w')
    valwfile = open(output + "val.txt",'w')
    datanum = len(result)
    for i in xrange(datanum):
        if (i < datanum*0.8):
            twfile.writelines(result[i])
        else:
            valwfile.writelines(result[i])


def mat2txtlabeldata(inputmat,outputtxt):
    mat = scipy.io.loadmat(inputmat)
    filelist=mat['celebrityImageData'][0][0][1]
    labellist =mat['celebrityImageData'][0][0][7]
    file = open(os.path.abspath('.')+'/'+outputtxt,'w')
    count=0
    for l,d in zip(filelist,labellist):
        count += 1
        #print count,str(d[0][0]), str(l[0])
        outstr='/'+d[0][0]+" "+ str(l[0])+"\n"
        #print outstr
        outstr=outstr.encode('UTF-8')
        file.writelines(outstr)
if __name__=="__main__":
    if (len(sys.argv)!=3):
        sys.exit("Usage:python {0} inputmat outputtxt".format(sys.argv[0]))
mat2txtlabeldata(sys.argv[1],sys.argv[2])
shufuldata(sys.argv[2],os.path.abspath('.')+'/')

