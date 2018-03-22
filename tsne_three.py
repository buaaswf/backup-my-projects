# -*- coding: utf-8 -*-
__author__ = 'TT'
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np

from sklearn.cross_validation import train_test_split
from time import time
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import (manifold, datasets, decomposition, ensemble, lda,random_projection)
import os

with open('E:/file/nodules/scatter/output-fc1000-ytest.pickle','rb') as f:
    y_test = pickle.load(f)

with open('E:/file/nodules/scatter/output-dthreeway1025-model1-fc1000-11022.pickle','rb') as f:
    X_test = pickle.load(f)
    print X_test.shape

with open('E:/file/nodules/scatter/output-dthreeway1025-model2-fc1000-11022.pickle','rb') as f:
    X_lbp_test = pickle.load(f)
    print X_lbp_test.shape
with open('E:/file/nodules/scatter/output-dthreeway1025-model3-fc1000-11022.pickle','rb') as f:
    X_hog_test = pickle.load(f)
    print X_hog_test.shape

X = np.concatenate((X_test,X_lbp_test,X_hog_test),axis=1)

# with open('E:/file/nodules/scatter/output-orginal-test--fc1000-1007-13.pickle','rb') as f:
#     X = pickle.load(f)

# with open('E:/file/nodules/scatter/output-addlbphog6-test--fc1000-1007-13.pickle','rb') as f:
#     X = pickle.load(f)

print X.shape
#%%
# 将降维后的数据可视化,2维
def plot_embedding_2d(X, title=None):
    #坐标缩放到[0,1]区间
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)

    #降维后的坐标为（X[i, 0], X[i, 1]），在该位置画出对应的digits

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1],str(y_test[i]),
                 color=plt.cm.Set1(y_test[i]),
                 fontdict={'weight': 'bold', 'size': 5})

    if title is not None:
        plt.title(title)

#%%
#将降维后的数据可视化,3维
def plot_embedding_3d(X):
    #坐标缩放到[0,1]区间

    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)
    #降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
    fig = plt.figure()

    ax = fig.add_subplot(1, 1,1, projection='3d',)
    #ax.patch.set_facecolor('red')
    ax.patch.set_alpha(0.9)
    ax.set_zlim(0.1,0.9)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    #ax.set_axis_bgcolor('#FFFFFF')
    plt.xlim(0.2,1)
    plt.ylim(0.05,1)


    # edgecolors='black',
    ax.scatter(X[y_test==0][:,0], X[y_test==0][:,1], X[y_test==0][:,2],linewidths =1,edgecolors='black',color = 'red', label='benign',marker='o')
    ax.scatter(X[y_test==1][:,0], X[y_test==1][:,1], X[y_test==1][:,2],linewidths =1,edgecolors='black',color = 'blue', label='malignant',marker='o')
    # for i in range(X.shape[0]):
    #     ax.text(X[i, 0], X[i, 1], X[i,2],str(y_test[i]),
    #              color=plt.cm.Set1(y_test[i]),
    #              fontdict={'weight': 'bold', 'size': 6})



'''
iris = load_iris()

X_tsne = TSNE(learning_rate=100).fit_transform(iris.data)
X_pca = PCA().fit_transform(iris.data)



plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris.target)
plt.subplot(122)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target)
plt.show()
'''



#
#
X_tsne = TSNE(n_components=3, init='pca', random_state=0).fit_transform(X)

print X_tsne.shape

plot_embedding_3d(X_tsne)

plt.show()
'''

X_tsne = TSNE(random_state=0).fit_transform(X)
plt.figure()



p1 = plt.scatter(X_tsne[y_test==0][:,1], X_tsne[y_test==0][:,0], color = 'm', label='benign',marker='o')
p2 = plt.scatter(X_tsne[y_test==1][:,1], X_tsne[y_test==1][:,0], color = 'green', label='malignant',marker='o')
plt.legend(loc = 'upper right')
plt.show()


if os.path.exists('E:/file/nodules/scatter/tsne5-13-2000.pickle'):
	with open('E:/file/nodules/scatter/tsne5-13-2000.pickle','r') as rf:
		X_tsne = pickle.load(rf)
else:
	X_tsne = TSNE(learning_rate=2000, random_state=0).fit_transform(out)
	with open('E:/file/nodules/scatter/tsne5-13-2000.pickle','w') as wf:
		pickle.dump(X_tsne,wf)
'''
