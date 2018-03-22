#!/usr/bin/env python
# encoding: utf-8

from gen_path_label import gen_path_label
from get_face_fea import get_face_fea
from casia_wrapper import load_model, str_gen
from sklearn.metrics.pairwise import cosine_similarity
import cPickle as pickle
import sys
import math
from color_transfer import color_transfer
from color_transfer import image_stats
import cv2
import caffe


net, transformer = load_model()


def MahalanobisDistance(path_a, path_b):
    imagex, imagey = cv2.imread(path_a), cv2.imread(path_b)
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(imagex)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(imagey)
    distance = math.fabs(lMeanSrc - lMeanTar) + math.fabs(aMeanSrc - aMeanTar) + math.fabs(bMeanSrc - bMeanTar)
    return distance


def get_valid_feas(folder, do_pca=False):
    """
    given a floder, auto generate label according to its child folders,
    and return the valid face CNN features, labels, aligned picture paths
    """
    paths, labels = gen_path_label(folder)
    valid_feas, valid_labels, aligned_paths = [], [], []
    for path, label in zip(paths, labels):
        aligned_path, fea = get_face_fea(path, do_pca)
        if fea:
            valid_feas.append(fea)
            valid_labels.append(label)
            aligned_paths.append(aligned_path)
    return valid_feas, valid_labels, aligned_paths


def calculate_sim(valid_feas, valid_labels, aligned_paths, ma_dis_thrhd):
    """
    given features, labels, aligned face paths,
    return pair cosine_similarity of every two faces
    eg: [(sim, label_1, label_2, aligned_path_1, aligned_path_2), ... ]
    """
    assert(len(valid_feas) == len(valid_labels) == len(aligned_paths))

    trsfed_imgs_folder = '/home/s.li/web-g206/find-lost/app/static/transfered/'
    pair_results = []
    for i in xrange(len(valid_feas)):
        print("{0}/{1}".format(i, len(valid_feas)))
        for j in xrange(i + 1, len(valid_feas)):
            if MahalanobisDistance(aligned_paths[i], aligned_paths[j]) < ma_dis_thrhd:
                sim = cosine_similarity(valid_feas[i]['data'][0], valid_feas[j]['data'][0])
                pair_results.append((sim, valid_labels[i], valid_labels[j], aligned_paths[i], aligned_paths[j]))
            else:
                # color transfer
                trsfed_img_i = color_transfer(cv2.imread(aligned_paths[j]), cv2.imread(aligned_paths[i]))

                # save transfered image
                trsfed_img_fn = trsfed_imgs_folder + str_gen(size=8) + '.jpg'
                cv2.imwrite(trsfed_img_fn, trsfed_img_i)

                # get CNN feature of the transfered image
                caffe.set_device(0)
                caffe.set_mode_gpu()
                net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(trsfed_img_fn))
                net.forward()  # call once for allocation
                feat = net.blobs['ip3'].data[0]
                featline = feat.flatten()

                # calculate sim
                sim = cosine_similarity(featline.reshape(1, -1), valid_feas[j]['data'][0])
                pair_results.append((sim, valid_labels[i], valid_labels[j], trsfed_img_fn, aligned_paths[j]))

    return pair_results


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("usage python {0} folder output.pkl do_pca(0/1) ma_dis_threshold".format(sys.argv[0]))

    folder = sys.argv[1]
    pkl_fn = sys.argv[2]
    do_pca = int(sys.argv[3])
    ma_dis_thrld = int(sys.argv[4])
    print("getting valid feas...")
    valid_feas, valid_labels, aligned_paths = get_valid_feas(folder, do_pca)
    print("calculate sim...")
    pair_results = calculate_sim(valid_feas, valid_labels, aligned_paths, ma_dis_thrld)

    #  for pair in pair_results:
    #  print pair

    with open(pkl_fn, 'w') as f:
        pickle.dump(pair_results, f)
