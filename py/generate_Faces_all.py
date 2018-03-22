import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.grid_search import GridSearchCV
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
import traceback
plt.switch_backend('agg')
rng = np.random.RandomState(0)
from com_class_general import *
import os
import PIL
from PIL import Image
from skimage import data, img_as_float
# xsize = 100
# ysize = 100
xsize = 256
ysize = 256
def single_filter(image):
    dataimage =  np.asarray(image) 
    dataimage = img_as_float(dataimage)
    newimage = denoise_bilateral(image, sigma_range=0.05, sigma_spatial=5)
    im = PIL.Image.fromarray(newimage)
    return im
def readimages2array(dir = "./data/swf_clean/"):
    list_dirs = os.walk(dir)
    data_r = []
    data_g = []
    data_b = []
    for root,dirs,files in os.walk(dir):
        if not files:
            continue
        for d in files:
            try:
                imagefile = os.path.join(root,d)
                print imagefile
                img = PIL.Image.open(imagefile)
		img=np.asarray(img)
		img = cv2.resize(img,(ysize,xsize),interpolation=cv2.INTER_CUBIC)
		img = Image.fromarray(img, 'RGB')
		
		print "=====",(img)
		#print img

		#img.load()
                #img2 = img.resize((ysize,xsize),Image.ANTIALIAS)
		img.thumbnail((ysize,xsize), Image.ANTIALIAS)
		print img.size
		#print img2
                img2 =img
		print img2.getdata()
                img2.load()
                r, g, b = img2.split()
                data_r.append(np.array(r).reshape(xsize*ysize))
                data_g.append(np.array(g).reshape(xsize*ysize))
                data_b.append(np.array(b).reshape(xsize*ysize))
            except Exception, e:
                traceback.print_exc()
                print str(e)
    return np.array(data_r), np.array(data_g), np.array(data_b)#,label
data_r,data_g,data_b = readimages2array()
# use grid search cross-validation to optimize the bandwidth
def generateRandomdata(data,componets):
    svd = TruncatedSVD(n_components = componets)
    data = svd.fit_transform(data)
    params = {'bandwidth':np.logspace(-1,1,20)}
    grid = GridSearchCV(KernelDensity(),params)
    grid.fit(data)
    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    kde = grid.best_estimator_
    new_data = kde.sample(5000, random_state = 0)
    new_data = svd.inverse_transform(new_data)
    # print new_data
    minval = np.min(new_data)
    maxval = np.max(new_data)
    # new_data = new_data - minval
    # newdata = newdata>0
    new_data[ new_data<0 ]=0
    new_data[ new_data>255 ]=255
    # new_data /= maxval -minval
    # new_data *= 255
    # print new_data
    return new_data
'''
X = data_r
params = {'bandwidth': np.logspace(-1, 1, 20)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit(X)
print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
# use the best estimator to compute the kernel density estimate
kde = grid.best_estimator_
# sample 44 new points from the data
new_data = kde.sample(100, random_state=0)
#new_data =svd.inverse_transform(new_data)
print new_data.shape
sio.savemat("kde.mat",{'kde_data': new_data})
# turn data into a 4x11 grid
new_data = new_data.reshape((100,141*165))#???200,1 or 1 200
#real_data = digits.data[:44].reshape((4, 11, -1))
sio.savemat("kde.mat",{'kde_data': new_data})
real_data = new_data
print real_data.shape
'''
#X = PIL.Image.merge('RGB',(data_r,data_g,data_b))
#real_data_g = new_data
#print new_data.shape
# plot real digits and resampled digits
def mergergbsequence(d_r,d_g,d_b):
    X=[PIL.Image.merge('RGB',(PIL.Image.fromarray(r.reshape(xsize,ysize).astype('uint8')),PIL.Image.fromarray(g.reshape(xsize,ysize).astype('uint8')),PIL.Image.fromarray(b.reshape(xsize,ysize).astype('uint8')))) for r, g, b in zip(d_r, d_g, d_b)]
    return X
def readirs(gdir,dstdir,componets,top):
    import os
    import traceback
    for root ,dirs ,files in os.walk(gdir):
        if not dirs:
            continue
        for d in dirs:
            pathdir = os.path.join(root,d)
            files = list(os.walk(pathdir))[0][-1]
            #print pathdir
            #try:
            saveimages(pathdir, dstdir + str(d), files,componets,top)
            #except Exception, e:
            #    traceback.print_exc()
            #    print str(e)
            #    continue

def saveimages(src,dst,files,componets, top = True):
    #print  files
    data_r,data_g,data_b = readimages2array(src)
    # print data_r
    real_data_r = generateRandomdata(data_r,componets)
    #print real_data_r[real_data_r!=data_r]
    real_data_g = generateRandomdata(data_g,componets)
    real_data_b = generateRandomdata(data_b,componets)
    X = mergergbsequence(real_data_r,real_data_g,real_data_b)
    if not os.path.exists(dst):
        os.makedirs(dst)
    #print"x", X
    print dst
    #print "file", files
    i = 0
    for inimage ,filename in zip(X,files):
        # print type(inimage)
        # outimg = denoise_bilateral(np.array(inimage), sigma_range=0.05, sigma_spatial=1)
        # im = PIL.Image.fromarray(outimg)
        # print filename
        i += 1
	if top==True:
            inimage.save(dst+'/'+"gen_"+str(filename))
	else:
	    outimage=single_filter(inimage)
	    #outimg = denoise_bilateral(np.array(inimage), sigma_range=0.05, sigma_spatial=1)
            #im = PIL.Image.fromarray(outimg)
            outimage.save(dst+'/'+"gen_"+str(filename))

    '''
    fig, ax = plt.subplots(10, 10, subplot_kw=dict(xticks=[], yticks=[]))
    for j in range(10):
        ax[0, j].set_visible(False)
        for i in range(5):
            im = ax[i, j].imshow(real_data[i*10+j],cmap='jet', interpolation='nearest')
            im.set_clim(0, 16)
            im = ax[i + 5, j].imshow(real_data[i*10+j],cmap='jet', interpolation='nearest')
            im.set_clim(0, 16)

    ax[0, 5].set_title('Selection from the input data')
    ax[5, 5].set_title('"New" digits drawn from the kernel density model')
    plt.savefig("pde.png")
    '''
# readirs("/home/g206/data/swfface/data/todo/train_casia_le80/","/home/g206/data/swfface/testjoint/data/done/train_casia_le80_100/",100)
# readirs("/home/g206/data/swfface/data/todo/train_casia_le80/","/home/g206/data/swfface/testjoint/data/done/train_casia_le80_15/",15)
# readirs("/home/g206/data/swfface/data/todo/train_casia_le80/","/home/g206/data/swfface/testjoint/data/done/train_casia_le80_60/",60)
# readirs("/home/g206/data/swfface/data/todo/val_casia_le20/","/home/g206/data/swfface/testjoint/data/done/val_casia_le20_60/",60)
# readirs("/home/g206/data/swfface/data/todo/val_casia_le20/","/home/g206/data/swfface/testjoint/data/done/val_casia_le20_15/",15)
# readirs("/home/g206/data/swfface/data/todo/val_casia_le20/","/home/g206/data/swfface/testjoint/data/done/val_casia_le20_100/",100)
# readirs("/home/g206/data/swfface/data/todo/train_casia_le80/","/home/g206/data/swfface/testjoint/data/done/train_casia_le80_100/",100)

# readirs("/home/g206/data/swfface/testjoint/data/todo/numless40/train","/home/g206/data/swfface/testjoint/data/done/train_casia_40_100/",100)
# readirs("/home/g206/data/swfface/testjoint/data/todo/numless40/train","/home/g206/data/swfface/testjoint/data/done/train_casia_40_15/",15)
# readirs("/home/g206/data/swfface/testjoint/data/todo/numless40/train","/home/g206/data/swfface/testjoint/data/done/train_casia_40_60/",60)

# readirs("/home/g206/data/swfface/testjoint/data/todo/numless40/val","/home/g206/data/swfface/testjoint/data/done/val_casia_40_100/",100)
# readirs("/home/g206/data/swfface/testjoint/data/todo/numless40/val","/home/g206/data/swfface/testjoint/data/done/val_casia_40_15/",15)
# readirs("/home/g206/data/swfface/testjoint/data/todo/numless40/val","/home/g206/data/swfface/testjoint/data/done/val_casia_40_60/",60)
# readirs("/home/s.li/caffe1128/data/cifa10/train/","/home/s.li/cifa10/cifa10generate/",900)
# readirs("./train/","traing/",800)
# readirs("/new_home/swf/CASIA-maxpy-clean_small/","/new_home/swf/CASIA-maxpy-clean_small_gen/",50)
# readirs("/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_0/","/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_0_gen2/",900)
# readirs("/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_1/","/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_1_gen2/",900)
# readirs("/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_2/","/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_2_gen2/",900)
# readirs("/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_3/","/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_3_gen2/",900)
# readirs("/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_4/","/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_4_gen2/",900)
# readirs("/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_5/","/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_5_gen2/",900)
# readirs("/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_6/","/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_6_gen2/",900)
# readirs("/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_7/","/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_7_gen2/",900)
# readirs("/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_8/","/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_8_gen2/",900)
# readirs("/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_9/","/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_9_gen2/",900)
# readirs("/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_10/","/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_10_gen2/",900)
# readirs("/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_11/","/new/swf/code/cifar10/cifar-10-batches-py/oversmapling_11_gen2/",900)
# readirs("/home/g206/data/cifa10/train","/home/g206/data/cifa10_900/",900)
# readirs("/home/g206/data/cifa10/train","/home/g206/data/cifa10_900/",900)
# readirs("/home/g206/data/cifa10/train","/home/g206/data/cifa10_150/",150)
# readirs("/home/g206/data/cifa10/train","/home/g206/data/cifa10_300/",300)
#readirs("/new/swf/code/cifar10/cifar-10-batches-py/cifar100/0520/oversample/h1/","/new/swf/code/cifar10/cifar-10-batches-py/cifar100/0520/gen/h1/",900)
#readirs("/new/swf/code/cifar10/cifar-10-batches-py/cifar100/0520/oversample/h3/","/new/swf/code/cifar10/cifar-10-batches-py/cifar100/0520/gen/h3/",900)
#readirs("/new/swf/code/cifar10/cifar-10-batches-py/cifar100/0520/oversample/h2/","/new/swf/code/cifar10/cifar-10-batches-py/cifar100/0520/gen/h2/",900)
#readirs("/new/swf/code/cifar10/cifar-10-batches-py/cifar100/0520/oversample/t11/","/new/swf/code/cifar10/cifar-10-batches-py/cifar100/0520/gen/t11/",900)
#readirs("/new/swf/code/cifar10/cifar-10-batches-py/cifar100/0520/oversample/t12/","/new/swf/code/cifar10/cifar-10-batches-py/cifar100/0520/gen/t12/",900)
#readirs("/new/swf/code/cifar10/cifar-10-batches-py/cifar100/0520/oversample/t13/","/new/swf/code/cifar10/cifar-10-batches-py/cifar100/0520/gen/t13/",900)
#readirs("/new/swf/code/cifar10/cifar-10-batches-py/cifar100/0520/oversample/t21/","/new/swf/code/cifar10/cifar-10-batches-py/cifar100/0520/gen/t21/",900)
#readirs("/new/swf/code/cifar10/cifar-10-batches-py/cifar100/0520/oversample/t22/","/new/swf/code/cifar10/cifar-10-batches-py/cifar100/0520/gen/t22/",900)
#readirs("/new/swf/code/cifar10/cifar-10-batches-py/cifar100/0520/oversample/t23/","/new/swf/code/cifar10/cifar-10-batches-py/cifar100/0520/gen/t23/",900)
#readirs("/new/swf/code/cifar10/cifar-10-batches-py/cifar100/0520/oversample/t23/","/new/swf/code/cifar10/cifar-10-batches-py/cifar100/0520/gen/t23/",900)
#readirs("/home/s.li/2017/data/png_oversmapling/oversmapling_0/2/","/home/s.li/2017/data/png_oversmapling_gen/oversmapling_0/2/",900)
#readirs("/home/s.li/2017/data/png_oversmapling/oversmapling_0/6/","/home/s.li/2017/data/png_oversmapling_gen/oversmapling_0/6/",900)
#readirs("/home/s.li/2017/data/png_oversmapling/oversmapling_0/8/","/home/s.li/2017/data/png_oversmapling_gen/oversmapling_0/8/",900)
#readirs("/home/s.li/2017/data/png_oversmapling/oversmapling_1/2/","/home/s.li/2017/data/png_oversmapling_gen/oversmapling_0/2/",900)
#readirs("/home/s.li/2017/data/png_oversmapling/oversmapling_1/4/","/home/s.li/2017/data/png_oversmapling_gen/oversmapling_0/4/",900)
#readirs("/home/s.li/2017/data/png_oversmapling/oversmapling_1/8/","/home/s.li/2017/data/png_oversmapling_gen/oversmapling_0/8/",900)
#readirs("/home/s.li/2017/data/png_oversmapling/oversmapling_1/4/","/home/s.li/2017/data/png_oversmapling_gen/oversmapling_0/4/",900)
#readirs("/home/s.li/2017/data/png_oversmapling/oversmapling_1/5/","/home/s.li/2017/data/png_oversmapling_gen/oversmapling_0/5/",900)
#readirs("/home/s.li/2017/data/png_oversmapling/oversmapling_1/6/","/home/s.li/2017/data/png_oversmapling_gen/oversmapling_0/6/",900)
#readirs("/home/s.li/2017/data/png_oversmapling/oversmapling_2/","/home/s.li/2017/data/png_oversmapling_gen/oversmapling_2/",900)
#readirs("/home/s.li/2017/data/png_oversmapling/oversmapling_3/","/home/s.li/2017/data/png_oversmapling_gen/oversmapling_3/",900)
#readirs("/home/s.li/2017/data/png_oversmapling/oversmapling_4/","/home/s.li/2017/data/png_oversmapling_gen/oversmapling_4/",900)
#readirs("/home/s.li/2017/data/png_oversmapling/oversmapling_5/","/home/s.li/2017/data/png_oversmapling_gen/oversmapling_5/",900)
#readirs("/home/s.li/2017/data/png_oversmapling/oversmapling_6/","/home/s.li/2017/data/png_oversmapling_gen/oversmapling_6/",900)
#readirs("/home/s.li/2017/data/png_oversmapling/oversmapling_7/","/home/s.li/2017/data/png_oversmapling_gen/oversmapling_7/",900)
#readirs("/home/s.li/2017/data/png_oversmapling/oversmapling_8/","/home/s.li/2017/data/png_oversmapling_gen/oversmapling_8/",900)
#readirs("/home/s.li/2017/data/png_oversmapling/oversmapling_9/","/home/s.li/2017/data/png_oversmapling_gen/oversmapling_9/",900)
#readirs("/home/s.li/2017/data/png_oversmapling/oversmapling_10/","/home/s.li/2017/data/png_oversmapling_gen/oversmapling_10/",900)
#readirs("/home1/data/imagenet/small_dataset/","/home1/data/imagenet/small_dataset_gen/",40000)
#readirs("/home1/data/imagenet/small_dataset/","/home1/data/imagenet/small_dataset_gen_10000/",10000)
#readirs("/home1/data/imagenet/small_dataset/","/home1/data/imagenet/small_dataset_gen_500/",200)
#readirs("/home1/data/imagenet/google_small_class_three/top/","/home1/data/imagenet/google_small_class_three/top_gen/",200,top=True)
#readirs("/home1/data/imagenet/google_small_class_three/last/","/home1/data/imagenet/google_small_class_three/last_gen/",300,top=False)
readirs("/home1/data/imagenet/residualnet_small_class_three/top/","/home1/data/imagenet/residualnet_small_class_three/top_gen/",200,top=True)
readirs("/home1/data/imagenet/residualnet_small_class_three/last/","/home1/data/imagenet/residualnet_small_class_three/last_gen/",300,top=False)
