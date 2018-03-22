"""
=======================================
Receiver Operating Characteristic (ROC)
=======================================

Example of Receiver Operating Characteristic (ROC) metric to evaluate
classifier output quality.

ROC curves typically feature true positive rate on the Y axis, and false
positive rate on the X axis. This means that the top left corner of the plot is
the "ideal" point - a false positive rate of zero, and a true positive rate of
one. This is not very realistic, but it does mean that a larger area under the
curve (AUC) is usually better.

The "steepness" of ROC curves is also important, since it is ideal to maximize
the true positive rate while minimizing the false positive rate.

Multiclass settings
-------------------

ROC curves are typically used in binary classification to study the output of
a classifier. In order to extend ROC curve and ROC area to multi-class
or multi-label classification, it is necessary to binarize the output. One ROC
curve can be drawn per label, but one can also draw a ROC curve by considering
each element of the label indicator matrix as a binary prediction
(micro-averaging).

Another evaluation measure for multi-class classification is
macro-averaging, which gives equal weight to the classification of each
label.

.. note::

    See also :func:`sklearn.metrics.roc_auc_score`,
             :ref:`sphx_glr_auto_examples_model_selection_plot_roc_crossval.py`.

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("Agg")
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

params = {'legend.fontsize': 'xx-large',
          'axes.labelsize': 'xx-large',
          'axes.titlesize': 'xx-large',
          'xtick.labelsize': 'xx-large',
          'ytick.labelsize': 'xx-large'}
plt.rcParams.update(params)
def drawroc(labels,pred,out):

    print "roc=========================>>>>M<<<<<<<<<<<<<<<<"
# Import some data to play with
# Binarize the output
    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target
    labels = label_binarize(labels, classes=[0, 1, 2,3,4,5,6,7,8,9])
    # print y
    y = np.array(labels)
    X = np.array(pred)
    # y = label_binarize(y, classes=[0, 1])
    print "-----------------------",y.shape
    n_classes = y.shape[1]
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # import pickle
    from sklearn import preprocessing
    from sklearn.preprocessing import OneHotEncoder
    #  file = open(pred,'rb')
    y_score_onehot = np.asarray(pred)
    # enc = preprocessing.OneHotEncoder()


    y_test=labels
    y_test=np.asarray(y_test)
    # from sklearn import preprocessing
    # y_test = label_binarize(y_test, classes=[0, 1])
    # n_classes = y_test.shape[1]
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test_onehot=lb.transform(y_test)
    print "====>>", y_test_onehot.shape
    print y_score_onehot.shape


    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    print "======>>",n_classes
    for i in range(n_classes):
        print "---->>", i
        #  fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        print y_test_onehot[:, i]
        print y_test_onehot[:, i].shape
        # print y_score_onehot[:, i].shape
        fpr[i], tpr[i], _ = roc_curve(y_test_onehot[:, i], y_score_onehot[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_onehot.ravel(), y_score_onehot.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


##############################################################################
# Plot of a ROC curve for a specific class
    plt.figure()
    lw = 1.8
    # plt.plot(fpr[2], tpr[2], color='darkorange',
            # lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate',fontsize=20)
    # plt.ylabel('True Positive Rate',fontsize=20)
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.savefig("test0.png")


##############################################################################
# Plot ROC curves for the multiclass problem

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=2)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=2)

    # colors = cycle(['red','black','aqua', 'darkorange', 'cornflowerblue'])
    colors = cycle(['red','black','aqua', 'darkorange', 'cornflowerblue',"blue","purple","green","yellow","gray"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    # plt.rc('xtick', labelsize=20)
    # plt.rc('ytick', labelsize=20)
    plt.xlim([0, 0.1])
    plt.ylim([0.8, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('single class accuracy of svhn dataset')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(out)
