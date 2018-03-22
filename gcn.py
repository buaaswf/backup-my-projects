import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def global_contrast_normalize(X, scale=1., subtract_mean=True, use_std=False, sqrt_bias=0., min_divisor=1e-8):
    """
    Global contrast normalizes by (optionally) subtracting the mean
    across features and then normalizes by either the vector norm
    or the standard deviation (across features, for each example).
    Parameters
    ----------
    X : ndarray, 2-dimensional
        Design matrix with examples indexed on the first axis and \
        features indexed on the second.
    scale : float, optional
        Multiply features by this const.
    subtract_mean : bool, optional
        Remove the mean across features/pixels before normalizing. \
        Defaults to `True`.
    use_std : bool, optional
        Normalize by the per-example standard deviation across features \
        instead of the vector norm. Defaults to `False`.
    sqrt_bias : float, optional
        Fudge factor added inside the square root. Defaults to 0.
    min_divisor : float, optional
        If the divisor for an example is less than this value, \
        do not apply it. Defaults to `1e-8`.
    Returns
    -------
    Xp : ndarray, 2-dimensional
        The contrast-normalized features.
    Notes
    -----
    `sqrt_bias` = 10 and `use_std = True` (and defaults for all other
    parameters) corresponds to the preprocessing used in [1].
    References
    ----------
    .. [1] A. Coates, H. Lee and A. Ng. "An Analysis of Single-Layer
       Networks in Unsupervised Feature Learning". AISTATS 14, 2011.
       http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
    """
    assert X.ndim == 2, "X.ndim must be 2"
    scale = float(scale)
    assert scale >= min_divisor

    # Note: this is per-example mean across pixels, not the
    # per-pixel mean across examples. So it is perfectly fine
    # to subtract this without worrying about whether the current
    # object is the train, valid, or test set.
    mean = X.mean(axis=1)
    if subtract_mean:
        X = X - mean[:, numpy.newaxis]  # Makes a copy.
    else:
        X = X.copy()

    if use_std:
        # ddof=1 simulates MATLAB's var() behaviour, which is what Adam
        # Coates' code does.
        ddof = 1

        # If we don't do this, X.var will return nan.
        if X.shape[1] == 1:
            ddof = 0

        normalizers = numpy.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
    else:
        normalizers = numpy.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale

    # Don't normalize by anything too small.
    normalizers[normalizers < min_divisor] = 1.

    X /= normalizers[:, numpy.newaxis]  # Does not make a copy.
    print np.asarray(X)
    return np.asarray(X)

def applyGCN(save_image=False):
    data = np.loadtxt("../inputData/train.csv", dtype=np.float32, delimiter=',', skiprows=1)
    test_data = np.loadtxt("../inputData/test.csv", dtype=np.float32, delimiter=',', skiprows=1)

    trainingFolder = "../inputData/converted_training/GCN/"
    testingFolder = "../inputData/converted_testing/GCN/"


    aux_data = data.copy()
    ###################################################
    #############TRAIN_DATA############################
    ###################################################
    # GCN#
    data_to_gcn = data[:, 1:]

    img_gcn = global_contrast_normalize(data_to_gcn)


    for x in xrange(0, len(data[:, 1])):
        print "Saving training image :", x

        image_aux = np.copy(img_gcn[x, :])

        image_aux = (image_aux).astype('uint8') * 255

        aux_data[x, 1:] = image_aux.tolist()

        if save_image is True:
            mpimg.imsave(trainingFolder + "GCN_Gray_TrainImage_" + str(x), image_aux, cmap=plt.get_cmap('gray'))
            #    figsize = [x / float(dpi) for x in (arr.shape[1], arr.shape[0])]
            #    IndexError: tuple index out of range

    with open(trainingFolder + "GCN_traindata.csv", 'wb') as fp:
        for i in range(0, aux_data.shape[0]):
            column = aux_data[i, :].tolist()
            column = map(lambda x: str(x) + ',', column)
            column = ''.join(column)[0:-1]
            fp.write(column + '\n')

    ##################################################
    #############TEST_DATA############################
    ##################################################
    img_gcn = global_contrast_normalize(test_data[:, :])
    aux_data = test_data.copy()

    for x in xrange(0, len(test_data[:, 0])):
        print "Saving testing image :", x

        image_aux = np.copy(img_gcn[x, :])
        aux_data[x, :] = image_aux.tolist()

        image_aux = np.reshape(image_aux, (28, 28))

        if save_image is True:
            mpimg.imsave(testingFolder + "GCN_Gray_TestImage_" + str(x), image_aux, cmap=plt.get_cmap('gray'))

    with open(testingFolder + "GCN_testdata.csv", 'wb') as fp:
        for i in range(0, aux_data.shape[0]):
            column = aux_data[i, :].tolist()
            column = map(lambda x: str(x) + ',', column)
            column = ''.join(column)[0:-1]
            fp.write(column + '\n')

if __name__=="__main__":

    image= cv2.imread("gen_n02025239_58.JPEG")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print  gray_image
    cv2.imshow("blue",global_contrast_normalize(gray_image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()