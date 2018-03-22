#!/usr/bin/env python
# encoding: utf-8

import socket
from retrieval.load_data import get_files
import numpy as np
import Image
import os
import cPickle as pickle
from sklearn.metrics.pairwise import cosine_similarity
from face_verify_client import send
from scipy.io import loadmat
from config import CNN_FEA_HOST, CNN_FEA_PORT, ALIGN_HOST, ALIGN_PORT


def send2align(src_path):
    """
    @params: src_path, the face image path to be detected and aligned
    @return: return the aligned face image path, if failed, return fail info, eg
        error info and 'no face found'
    """
    HOST, PORT = ALIGN_HOST, ALIGN_PORT
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        sock.sendall(src_path)
        recv_data = sock.recv(1024)
        # print "recv: {0}".format(recv_data)
        sock.close()
        return recv_data
    except Exception, e:
        # print "got an error {0}".format(e)
        sock.close()
        return "error {0}".format(e)


def path2arr(path):
    """
    read image from path, return numpy array,
    return empty array if failed
    """
    try:
        img = Image.open(path)
        img = img.resize((47, 55))
        # return np.array(img, dtype="int")
        return img
    except:
        return None


def valid_img_path(path):
    """
    check if the path is valid
    return bool
    """
    try:
        Image.open(path)
        return True
    except:
        return False


def get_cnn_fea(aligned_paths):
    """
    transform aligned face images to CNN features
    """
    print "Getting cnn fea."
    print aligned_paths
    fea_host, fea_port = CNN_FEA_HOST, CNN_FEA_PORT
    fea_paths = [send(path + ';1', fea_host, fea_port) for path in aligned_paths]
    features = []
    for path in fea_paths:
        mat = loadmat(path)
        features.append(np.asarray(mat['data'], dtype='float'))
    return features


class DataBase:
    def __init__(self, data_path):
        self.data_path = data_path
        self.batch_size = 1000
        self.data = []
        self.loadData()

    def loadData(self):
        file_names = get_files(self.data_path)
        data = []
        for fn in file_names:
            with open(fn, 'r') as f:
                data += pickle.load(f)
        self.data = data

    def __save2database(self, imgIDs, cnn_feas):
        """
        save data to database(pickle files)
        save style: [(id_1, fea_1), .... , (id_batch_size, fea_batch_size)]
        """
        data_folder = self.data_path
        batch_size = self.batch_size

        def save(count, imgIDs, cnn_feas):
            fn = data_folder + count.__str__() + '.pkl'
            if os.path.isfile(fn):
                with open(fn, 'r') as f:
                    orgnl_data = pickle.load(f)
                    left_space = batch_size - len(orgnl_data)
                with open(fn, 'w') as f:
                    if left_space >= len(imgIDs):
                        pickle.dump(orgnl_data + zip(imgIDs, cnn_feas), f)
                    else:
                        pickle.dump(orgnl_data + zip(imgIDs[0:left_space], cnn_feas[0:left_space]), f)
                        save(count + 1, imgIDs[left_space::], cnn_feas[left_space::])
            else:
                with open(fn, 'w') as f:
                    if batch_size >= len(imgIDs):
                        pickle.dump(zip(imgIDs, cnn_feas), f)
                    else:
                        pickle.dump(zip(imgIDs[0:batch_size], cnn_feas[0:batch_size]), f)
                        save(count + 1, imgIDs[batch_size::], cnn_feas[batch_size::])

        file_names = get_files(data_folder)
        n_files = len(file_names)
        count = n_files if n_files != 0 else 1
        save(count, imgIDs, cnn_feas)

    def add_faces(self, imgIDs, img_paths):
        """
        model: retrieval_face.Model
        add faces to database,
        return invalid image ids
        """
        print "Adding faces."
        inval_ids = []
        aligned_paths = []
        for path, id in zip(img_paths, imgIDs):
            aligned_path = send2align(path)
            if not valid_img_path(aligned_path):
                inval_ids.append(id)
                imgIDs.remove(id)
            else:
                aligned_paths.append(aligned_path)
        if len(aligned_paths) != 0:
            cnn_feas = get_cnn_fea(aligned_paths)
            self.__save2database(imgIDs, cnn_feas)
        return inval_ids

    def search(self, fn, top_n=10, sim_thresh=None):
        """
        retrieval face from database,
        return top_n similar faces' imgIDs, return None if failed
        """
        print "\n\nsearch...\n\n"
        if top_n > len(self.data):
            top_n = len(self.data)
        aligned_fn = send2align(fn)
        if not valid_img_path(aligned_fn):
            print "align none."
            return None

        cnn_fea = get_cnn_fea([aligned_fn])[0]

        # print "cnn_fea: {0}".format(cnn_fea[0])
        sims = [cosine_similarity(cnn_fea[0], item[1][0])[0][0] for item in self.data]
        # print len(self.data), len(sims)
        # for i in range(len(sims)):
            # print sims[i], self.data[i][0]
        sort_index = np.argsort(-np.array(sims))
        result = []
        print sort_index
        if sim_thresh is None:
            for index in np.nditer(sort_index):
                cur_id = self.data[index][0].split('-')[0]
                if cur_id not in result and len(result) < top_n:
                    result.append(cur_id)
            return result
        else:
            for index in np.nditer(sort_index):
                if sims[index] < sim_thresh:
                    break
                cur_id = self.data[index][0].split('-')[0]
                if cur_id not in result:
                    result.append(cur_id)
            return result


if __name__ == '__main__':
    data_path = './temp/'
    dataBase = DataBase(data_path)
    test_fns = [
        '/home/g206/data/baidu/origin/man/1/58.jpg',
        '/home/g206/data/baidu/origin/man/1/53.jpg',
        '/home/g206/data/baidu/origin/man/1/51.jpg',
        '/home/g206/data/baidu/origin/man/1/57.jpg',
        '/home/g206/data/baidu/origin/man/1/14.jpg',
        '/home/g206/data/baidu/origin/man/1/16.jpg',
        '/home/g206/data/baidu/origin/man/1/34.jpg',
        '/home/g206/data/baidu/origin/man/1/25.jpg',
        '/home/g206/data/baidu/origin/man/1/22.jpg',
        '/home/g206/data/baidu/origin/man/1/21.jpg',
    ]
    test_ids = [i.__str__() + '-1' for i in range(len(test_fns))]

    # dataBase.add_faces(test_ids, test_fns)
    dataBase.loadData()

    search_result = dataBase.search(test_fns[1])
    print search_result

    search_result = dataBase.search(test_fns[0], sim_thresh=0.0)
    print search_result
