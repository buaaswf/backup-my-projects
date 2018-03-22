import pandas as pd
from pandas import DataFrame
import os
import shutil
#data = pd.read_csv('imagenet_alex_dvn.txt', sep=' ',skipinitialspace =True)
#data = pd.read_csv('imagenet_alex_dvn.txt',delimiter=' ')
def copytree(src, dst, symlinks=False):  
    names = os.listdir(src)  
    if not os.path.isdir(dst):  
        os.makedirs(dst)  
          
    errors = []  
    for name in names:  
        srcname = os.path.join(src, name)  
        dstname = os.path.join(dst, name)  
        try:  
            if symlinks and os.path.islink(srcname):  
                linkto = os.readlink(srcname)  
                os.symlink(linkto, dstname)  
            elif os.path.isdir(srcname):  
                copytree(srcname, dstname, symlinks)  
            else:  
                if os.path.isdir(dstname):  
                    os.rmdir(dstname)  
                elif os.path.isfile(dstname):  
                    os.remove(dstname)  
                shutil.copy2(srcname, dstname)  
        except (IOError, os.error) as why:  
            errors.append((srcname, dstname, str(why)))  
        # catch the Error from the recursive copytree so that we can  
        # continue with other files  
        except OSError as err:  
            errors.extend(err.args[0])  
    try:  
        copystat(src, dst)  
    except WindowsError:  
        # can't copy file access times on Windows  
        pass  
    except OSError as why:  
        errors.extend((src, dst, str(why)))  
    if errors:  
        raise Error(errors) 

def gettoplist():
    #data = pd.read_csv('imagenet_alex_dvn.txt', sep='\s+',error_bad_lines=False)
    #data = pd.read_csv('imagenet_googlenet_dvn.txt', sep='\s+',error_bad_lines=False)
    data = pd.read_csv('imagenet_vggnet_dvn.txt', sep='\s+',error_bad_lines=False)
    #print data.header

    #df = DataFrame(data)
    print data.columns
    print data
    #print data.index.name
    sortdata = data.sort_values(by=['precision'], ascending = False)
    #print sortdata
    list1 = sortdata['precision'].tolist()
    #print list1.tolist()
    list2 = sortdata['id'].tolist()
    count = -1
    toplist=[]
    lastlist=[]
    normallist = []
    for pre, classid in zip(list1,list2):
	if pre > list1[int(0.3*len(list1))]:
	    print pre, list1[int(0.3*len(list1))]
	    toplist.append(classid)
	    print len(toplist)
	   
	elif pre < list1[int(0.7*len(list1))]:
	   print pre, list1[int(0.7*len(list1))]
	   lastlist.append(classid)
	else:
	    normallist.append(classid)
    return [toplist, lastlist, normallist]
def seperate_class(threelists, inpath, outpath):
    toplist = threelists[0]
    lastlist = threelists[1]
    normallist = threelists[2]
    print os.path.join(outpath,"top")
    if not os.path.exists(os.path.join(outpath,"top")):
	os.makedirs(os.path.join(outpath,"top"))
	os.makedirs(os.path.join(outpath,"last"))
	os.makedirs(os.path.join(outpath,"norm"))
    for top in toplist:
	print top
	shutil.copytree(os.path.join(inpath,str(top)), os.path.join(outpath,"top",str(top)) )
    for last in lastlist:
	shutil.copytree(os.path.join(inpath,str(last)), os.path.join(outpath, "last",str(last)) )
    for norm in normallist:
	shutil.copytree(os.path.join(inpath,str(norm)), os.path.join(outpath, "norm",str(norm)) )
    

if __name__=="__main__":
    res = gettoplist()
    seperate_class(res,"/home/user/swfdata/lmdb/small_dataset/train", "/home/user/swfdata/lmdb/ninnet_small_class_three/")
