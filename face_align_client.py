#!/usr/bin/env python
# encoding: utf-8

import socket
from retrieval.load_data import get_files
from retrieval_face import Model
import numpy as np
import Image
import os
import cPickle as pickle
from sklearn.metrics.pairwise import cosine_similarity


def send2align(src_path):
    """
    @params: src_path, the face image path to be detected and aligned
    @return: return the aligned face image path, if failed, return fail info, eg
        error info and 'no face found'
    """
    HOST, PORT = "localhost", 3200
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


class DataBase:
    def __init__(self, params_file, data_path):
        self.model = Model(params_file, batch_size=1)
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

    def __save2database(self, imgIDs, deepIDfeas):
        """
        save data to database(pickle files)
        save style: [(id_1, fea_1), .... , (id_batch_size, fea_batch_size)]
        """
        data_folder = self.data_path
        batch_size = self.batch_size

        def save(count, imgIDs, deepIDfeas):
            fn = data_folder + count.__str__() + '.pkl'
            if os.path.isfile(fn):
                with open(fn, 'r') as f:
                    orgnl_data = pickle.load(f)
                    left_space = batch_size - len(orgnl_data)
                with open(fn, 'w') as f:
                    if left_space >= len(imgIDs):
                        pickle.dump(orgnl_data + zip(imgIDs, deepIDfeas), f)
                    else:
                        pickle.dump(orgnl_data + zip(imgIDs[0:left_space], deepIDfeas[0:left_space]), f)
                        save(count + 1, imgIDs[left_space::], deepIDfeas[left_space::])
            else:
                with open(fn, 'w') as f:
                    if batch_size >= len(imgIDs):
                        pickle.dump(zip(imgIDs, deepIDfeas), f)
                    else:
                        pickle.dump(zip(imgIDs[0:batch_size], deepIDfeas[0:batch_size]), f)
                        save(count + 1, imgIDs[batch_size::], deepIDfeas[batch_size::])

        file_names = get_files(data_folder)
        n_files = len(file_names)
        count = n_files if n_files != 0 else 1
        save(count, imgIDs, deepIDfeas)

    def add_faces(self, imgIDs, img_paths):
        """
        model: retrieval_face.Model
        add faces to database,
        return invalid image ids
        """
        inval_ids = []
        imgs = []
        for path, id in zip(img_paths, imgIDs):
            aligned_path = send2align(path)
            aligned_arr = path2arr(aligned_path)
            if aligned_arr is None:
                inval_ids.append(id)
                imgIDs.remove(id)
            else:
                imgs.append(aligned_arr)
        if len(imgs) != 0:
            deepIDfeas = self.model.getID(imgs)
            self.__save2database(imgIDs, deepIDfeas)
        return inval_ids

    def search(self, fn, top_n=10, sim_thresh=None):
        """
        retrieval face from database,
        return top_n similar faces' imgIDs, return None if failed
        """
        if top_n > len(self.data):
            top_n = len(self.data)
        aligned_fn = send2align(fn)
        aligned_arr = path2arr(aligned_fn)
        if aligned_arr is None:
            print "align none."
            return None
        deepIDfea = self.model.getID([aligned_arr])[0]
        sims = [cosine_similarity(deepIDfea, item[1])[0][0] for item in self.data]
        #print len(self.data), len(sims)
        for i in range(len(sims)):
            print sims[i], self.data[i][0]
        sort_index = np.argsort(-np.array(sims))
        result = []
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
    params_file = '/home/dujunyi/work/faceretrieval/params_ourface.pkl'
    dataBase = DataBase(params_file)
    test_fns = [
        '/home/huangjiang/data/lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg',
        '/home/huangjiang/data/lfw-deepfunneled/Aaron_Guiel/Aaron_Guiel_0001.jpg',
        '/home/huangjiang/data/lfw-deepfunneled/Aaron_Patterson/Aaron_Patterson_0001.jpg',
        '/home/huangjiang/data/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg',
        '/home/huangjiang/data/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0002.jpg',
        '/home/huangjiang/data/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0003.jpg',
        '/home/huangjiang/data/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0004.jpg',
        '/home/huangjiang/data/lfw-deepfunneled/Aaron_Pena/Aaron_Pena_0001.jpg',
        '/home/huangjiang/data/lfw-deepfunneled/Aaron_Sorkin/Aaron_Sorkin_0001.jpg',
        '/home/huangjiang/data/lfw-deepfunneled/Aaron_Sorkin/Aaron_Sorkin_0002.jpg',
        '/home/huangjiang/data/lfw-deepfunneled/Aaron_Tippin/Aaron_Tippin_0001.jpg',
    ]
    test_ids = [i.__str__() + '-1' for i in range(len(test_fns))]

    dataBase.add_faces(test_ids, test_fns)
    dataBase.loadData()

    search_result = dataBase.search(test_fns[0])
    print search_result

    search_result = dataBase.search(test_fns[0], sim_thresh=0.0)
    print search_result
