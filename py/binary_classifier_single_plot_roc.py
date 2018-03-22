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
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("Agg")
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
print(__doc__)
params = {'legend.fontsize': 'xx-large',
          'axes.labelsize': 'xx-large',
          'axes.titlesize': 'xx-large',
          'xtick.labelsize': 'xx-large',
          'ytick.labelsize': 'xx-large'}


def drawroc(labels, pred, out):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    from itertools import cycle

    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    # from sklearn.model_selection import train_test_split
    # from sklearn.preprocessing import label_binarize
    # from sklearn.multiclass import OneVsRestClassifier
    # from scipy import interp

    y = labels

    y_score_onehot = np.array(pred)[:, 1]

    y_test_onehot = y

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    print y_test_onehot
    print y_score_onehot
    fpr, tpr, _ = roc_curve(y_test_onehot, y_score_onehot)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 3

    # Plot all ROC curves
    plt.figure()
    color = 'red'
    plt.plot(fpr, tpr, color=color, lw=lw,
             label='ROC curve of class (area = {0:0.4f})'
                   ''.format(roc_auc))

    plt.rcParams.update(params)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(out)

def singledrawroc(labels, pred, out,classname):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    from itertools import cycle

    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    # from sklearn.model_selection import train_test_split
    # from sklearn.preprocessing import label_binarize
    # from sklearn.multiclass import OneVsRestClassifier
    # from scipy import interp

    y = labels

    y_score_onehot = np.array(pred)[:, 1]

    y_test_onehot = y

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    print y_test_onehot
    print y_score_onehot
    fpr, tpr, _ = roc_curve(y_test_onehot, y_score_onehot)
    roc_auc = auc(fpr, tpr)

    # plt.figure()
    lw = 3

    # Plot all ROC curves

    color = 'red'
    plt.plot(fpr, tpr, color=color, lw=lw,
             label='ROC curve of class{1:s} (area = {2:0.4f})'
                   ''.format('',classname, roc_auc))

    plt.rcParams.update(params)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(out)


def batch_draw_roc(data):
    plt.figure()
    colors = ['red','blue','black','yellow','green','gray']
    i=-1
    out =  "batchroc_10"  + ".png"
    for (name, [labels, pred]) in data.items():
        i+=1
        # print labels,predroc

        # from sklearn.model_selection import train_test_split
        # from sklearn.preprocessing import label_binarize
        # from sklearn.multiclass import OneVsRestClassifier
        # from scipy import interp

        y = labels[0]

        y_score_onehot = np.array(pred[0])[:, 1]

        y_test_onehot = y

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        print y_test_onehot
        print y_score_onehot
        fpr, tpr, _ = roc_curve(y_test_onehot, y_score_onehot)
        roc_auc = auc(fpr, tpr)

        # plt.figure()
        lw = 3

        # Plot all ROC curves

        color = colors[i]
        plt.plot([0, 1], [0, 1], 'k--')
        # plt.plot(fpr, tpr, color=color, lw=lw,
        # plt.plot(fpr, tpr,  lw=lw,
                 # label='ROC curve of class{1:s} (area = {2:0.4f})'
                       # ''.format('', name, roc_auc))
        plt.plot(fpr, tpr,  lw=lw,
                 label='{1:s} (auc = {2:0.4f})'
                       ''.format('', name, roc_auc))

        plt.rcParams.update(params)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0, 0.45])
    plt.ylim([0.8, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(out)

        # singledrawroc(labels[0], predroc[0], "batchroc_10" + name + ".png",name)
