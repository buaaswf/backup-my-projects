#!/usr/bin/env python
# encoding: utf-8
import scipy.io as sio
import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
datalabel=sio.loadmat(r"matlab.mat")
for key in datalabel.keys():
    print key,type(datalabel[key])


data= datalabel['data']
label= datalabel['labels']
a_train, a_test, b_train, b_test = train_test_split(data, label, test_size=0.33, random_state=42)
clf = svm.LinearSVC()
clf.fit(a_train, b_train)

#clf.save("svm.model")
#dec = clf.decision_function()
#dec.shape[1] # 4 classes: 4*3/2 = 6
pre = clf.predict(a_test)
poly_svc = svm.SVC(kernel='poly', degree=3, C=1).fit(X, y)
poly_svc.fit(a_train,b_train)
print poly_svc.score(a_test,b_test)
print clf.score(a_test,b_test)
#print metrics.classification_report(b_test, pre)





