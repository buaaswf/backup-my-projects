#!/usr/bin/env python
# encoding: utf-8

print(__doc__)


# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
import scipy.io as sio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
import traceback
import pickle
import shutil
import os
h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]
namesv2 = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "DecisionTreeClassifier Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "LDA" ,"KNN",
         "Passive-Aggressive","Perceptron","Ridge","MultinomialNB","BernoulliNB",
         "GBDT","sgd_l1","sgd_l2","NearestCntroid"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    QDA(),

    ]

classifiersv2 = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    KNeighborsClassifier(n_neighbors=10),
    PassiveAggressiveClassifier(n_iter = 50),
    Perceptron(n_iter =50),
    RidgeClassifier(tol=1e-2,solver = 'lsqr'),
    MultinomialNB(alpha=.001),
    BernoulliNB(alpha=.001),
    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=1, loss='deviance'),
    SGDClassifier(alpha=.001, n_iter=50, penalty='l1'),
    SGDClassifier(alpha=.001, n_iter=50, penalty='l2'),
     NearestCentroid()








]
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)
#datasets = [make_moons(noise=0.3, random_state=0),
 #           make_circles(noise=0.2, factor=0.5, random_state=1),
 #           linearly_separable
 #           ]
# labeldata=sio.loadmat(r"./generatemat/casia.mat")
#labeldata=sio.loadmat("predlayermat/-3/thyroid_conv5.mat")
labeldata=sio.loadmat("predlayermat/-3/thyroid_conv5.mat")
labeldata=sio.loadmat("predlayermat/-3/lsthyroid_conv5_doctor.mat")
lstest=sio.loadmat("predlayermat/-3/lsthyroid_conv5_doctor.mat")
labeldata=sio.loadmat("predlayermat/-3/lsthyroid_conv5_doctor.mat")
labeldata=sio.loadmat("predlayermat/-3/lsthyroid_conv5thyroidnew.mat")
# lstest=sio.loadmat("predlayermat/openothyroid_conv5.mat")
lstest=sio.loadmat("predlayermat/-3/openthyroid_conv5-small.mat")
#lstest=sio.loadmat("predlayermat/openothyroid_conv5.mat")
labeldata=sio.loadmat("predlayermat/-3/lsthyroid_conv5thyroidnew.mat")
labeldata=sio.loadmat("predlayermat/-3/lsthyroidnew_conv5-small.mat")
#labeldata=sio.loadmat("predlayermat/-3/openthyroid_conv5-small.mat")
#lstest=sio.loadmat(r"predlayermat/-3/lsthyroid_conv5.mat")
file = open("lsthyroid_conv5id_pred_gt.pkl",'r')
idlist = pickle.load(file)
datasets = [labeldata]
test = lstest['data']
print test.shape
#test += 2
test_label = lstest['labels'][0]
figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds in datasets:
    # preprocess dataset, split into training and test part
    X =labeldata['data']
    # X += 100;
    y = labeldata['labels'][0]
    X= np.nan_to_num(X)
 #   X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    # x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    # y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    #                      np.arange(y_min, y_max, h))

    # just plot the dataset first
    # cm = plt.cm.RdBu
    # cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    # ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    # Plot the training points
    #ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    '''
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
   '''
    i += 1

    # iterate over classifiers
    plt.figure()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve Multiple Classifiers')
    plt.legend(loc='best')

    #test = X_test
    #test_label = y_test
    for name, clf in zip(namesv2, classifiersv2):
        try:
            #ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            test = np.nan_to_num(test)
            pred = clf.predict(test)
	    print "=================="
	    print classification_report(test_label, clf.predict(test)) 
	    print accuracy_score(test_label, pred)
            print pred
	   
	    with open(name+'CFPNet.pkl', 'wb') as fp:
    		pickle.dump(pred, fp)
	    errlist = []
	    for i in range(0, len(pred)):
		if pred[i]!=test_label[i]:
		    errlist.append(idlist[i])
	    errorname = os.path.join("error",name+"err")
	    if not os.path.exists(errorname):
		os.makedirs(errorname)
	    for err in errlist:
		shutil.copy(err, errorname)
	    #errlist =idlist[pred!=test_label]
	    with open(name+'CFPNeterrorid.pkl', 'wb') as fp:
    		pickle.dump(errlist, fp)
	    
	    #np.save("CFPNetpred.pkl")
            print  name
            print  score
            print classification_report(y_test, clf.predict(X_test))
	    text_file = open(name+".txt", "w")
     	    text_file.write(name +" model\n")
    	    text_file.write(classification_report(y_test,clf.predict(X_test)))
            y_pred_grd = clf.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_grd)

            roc_auc = auc(fpr, tpr)
            plt.xlim(0, 0.5)
            plt.ylim(0.5, 1)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr, tpr,
                     # label='ROC curve of class{1:s} (area = {2:0.4f})'
                     label='{1:s} (auc = {2:0.4f})'
                           ''.format('', name, roc_auc))
            # plt.plot(fpr, tpr, label=name)
            plt.legend(loc="lower right")
            plt.grid(True)

            # plt.plot(fpr_rf, tpr_rf, label=name)
            # plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
            # plt.plot(fpr_grd, tpr_grd, label='GBT')
            # plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')



            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, m_max]x[y_min, y_max].
          #  if hasattr(clf, "decision_function"):
            #    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
           # DecisionTreeClassifier:
             #   Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
           # Z = Z.reshape(xx.shape)
           # ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot also the training points
            #ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
            # and testing points
            #ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                       #alpha=0.6)

            #ax.set_xlim(xx.min(), xx.max())
            #ax.set_ylim(yy.min(), yy.max())
            #ax.set_xticks(())
            #ax.set_yticks(())
            #ax.set_title(name)
            #ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                   # size=15, horizontalalignment='right')
            i += 1
        except Exception as e:
            print e
	    traceback.print_exc()
            continue
    plt.show()
    plt.savefig('multipleclassroc.png')


figure.subplots_adjust(left=.02, right=.98)
plt.savefig("comclasses.png")
