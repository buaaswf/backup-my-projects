import os
import shutil
def fewfiles(dir):
    count = 0
    for root,dirs,files in os.walk(dir):
        filelength = len(files)
        if filelength!= 0:
            count = count +filelength
    print "the number of files under<%s> is:%d" %(dir,count)
    return count
#fewfiles("./swfface/testjoint/data/swf_clean/")
def getdirnames(dirpath,dst,threshold):
    print "start"
    print dirpath
    print dst
    for root,dirs,files in os.walk(dirpath):
        #print root,dirs,files
        if not dirs:
            #print dirs
            continue
        for d in dirs:
            path = os.path.join(root,d)
            print "running "
            imagenum = fewfiles(path)
            print imagenum
            if imagenum > threshold:
                if not os.path.exists(dst+"/"+d):
                    #print dst+d
                   # print path
                    shutil.copytree(path,dst+d)
                    #print "done"
def datasetfilenumberhistgram(dirpath):
    numberlist=[]
    for root,dirs,files in os.walk(dirpath):
        #print root,dirs,files
        if not dirs:
            #print dirs
            continue
        for d in dirs:
            path = os.path.join(root,d)
            print "running "
            imagenum = fewfiles(path)
            print imagenum
            numberlist.append([d,imagenum])
            # if imagenum > threshold:
                # if not os.path.exists(dst+"/"+d):
                    # #print dst+d
                   # # print path
                    # shutil.copytree(path,dst+d)
    #with open()
    f = open('output.txt', 'w')
    for item in numberlist:
        f.write("%s\n" % item)

    return numberlist


#getdirnames("/home/g206/data/casia/CASIA-WebFace/","/home/g206/data/swfface/testjoint/data/todo/",100)
#getdirnames("/home/g206/data/casia/CASIA-WebFace/","/home/g206/data/casia/CASIA_100+/",100)
datasetfilenumberhistgram("./casia/CASIA-WebFace")




