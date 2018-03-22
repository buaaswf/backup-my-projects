import sys
import os
import numpy as np
from conv_net.layers import *
from conv_net.deepid_class import *
from conv_net.deepid_generate import *
from retrieval.load_data import *
from retrieval.pre_process import *
from data_prepare.vectorize_img import *
import Image
import copy
import cv2

class Model(DeepIDGenerator):

    def __init__(self, params_file, nkerns=[20,40,60,80], n_hidden=160, acti_func=relu, img_size=(3, 55, 47), batch_size=1000):

        pd_helper = ParamDumpHelper(params_file)
        exist_params = pd_helper.get_params_from_file()
        if len(exist_params) != 0:
            exist_params = exist_params[-1]
        else:
            print 'error, no trained params'
            return
        DeepIDGenerator.__init__(self, exist_params)
        self.img_size = img_size
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.layer_params(nkerns, batch_size)
        self.build_layer_architecture(n_hidden, acti_func)

    def getID(self, images):
        image_vector_len = np.prod(self.img_size)
        arrs   = []
        batch_size = self.batch_size
        for img in images:
            arr_img = np.asarray(img, dtype='float64')
            # print arr_img.shape, image_vector_len
            arr_img = arr_img.transpose(2,0,1).reshape((image_vector_len, ))
            arrs.append(arr_img)
        arrs = np.asarray(arrs, dtype='float64')
        n_batch = len(arrs) / batch_size
        vector = np.zeros((len(arrs),self.n_hidden))
        for i in range(n_batch):
            x = arrs[ i*batch_size: (i+1)*batch_size]
            y = self.generator(x)
            vector[i*batch_size: (i+1)*batch_size,:] = y
        if n_batch * batch_size < len(arrs):
            x = arrs[n_batch*batch_size: ]
            y = self.generator(x)
            vector[n_batch*batch_size:] = y
        return vector

    def getID_given_mat(self, images):
        image_vector_len = np.prod(self.img_size)
        arrs   = []
        batch_size = self.batch_size
        for img in images:
            #arr_img = np.asarray(img, dtype='float64')
            arr_img = img.transpose(2,0,1).reshape((image_vector_len, ))
            arrs.append(arr_img)
        arrs = np.asarray(arrs, dtype='float64')
        n_batch = len(arrs) / batch_size
        vector = np.zeros((len(arrs),self.n_hidden))
        for i in range(n_batch):
            x = arrs[ i*batch_size: (i+1)*batch_size]
            y = self.generator(x)
            vector[i*batch_size: (i+1)*batch_size,:] = y
        if n_batch * batch_size < len(arrs):
            x = arrs[n_batch*batch_size: ]
            y = self.generator(x)
            vector[n_batch*batch_size:] = y

        return vector


def load_train(data_folder):
    file_names = get_files(data_folder)
    x, y, paths = load_data_xy_2(file_names)
    return x, y, paths

class Retrieval:

    def __init__(self, params_file, train_data_folder, str_sim_metric='cos', batch_size=1000):
        print '--------loading deepid model--------'
        self.net = Model(params_file, batch_size=batch_size)
        print '--------done--------'
        print '--------loading vectors--------'
        self.train_x, self.train_y, self.paths = load_train(train_data_folder)
        self.str_sim_metric = str_sim_metric
        print '--------done--------'

    def search(self, test_x, top_num=10):
        sim_metric_method = sim_metric_methods_set[self.str_sim_metric]
        train_x = self.train_x
        if self.str_sim_metric == 'cos':
            test_x, train_x = norm_data(test_x, train_x)
        print test_x.shape[1], train_x.shape[1]
        assert test_x.shape[1] == train_x.shape[1]
        query_sample_num = len(test_x)
        search_results = []
        for i in range(query_sample_num):
            sample = test_x[i]
            sim_result = sim_metric_method(sample, train_x)
            sort_index = np.argsort(sim_result)
            search_result = []
            for index in sort_index[0:top_num]:
                search_result.append((self.train_y[index], self.paths[index], sim_result[index]))
            search_results.append(search_result)
        return  search_results

    def get_result(self, images, top_num=10):
        test_x = self.net.getID(images)
        result = self.search(test_x, top_num)
        return (test_x, result)

    def get_result_given_mat(self, images, top_num=10):
        test_x = self.net.getID_given_mat(images)
        result = self.search(test_x, top_num)
        return (test_x, result)

def getimgs():
    folder_path = '/home/dujunyi/work/faceretrieval/'
    path_and_labels = read_csv_file('ourface_test.csv') + read_csv_file('ourface_train.csv')
    random.shuffle(path_and_labels)
    print path_and_labels[:10]
    images = []
    i = 0

    for (path, label) in path_and_labels:
        img = Image.open(folder_path + path)
        images.append(img)
        i = i + 1
        if i==10:
            break
    return images

def get_top10_given_mat(retrieval, mat):
    images = []
    images.append(mat)

    a, result = retrieval.get_result_given_mat(images)
    result = result[0]



    top10 = []
    for label, path, values in result:

        filename = '/home/dujunyi/work/faceretrieval/data/' + path
        print filename
        #img = cv2.imread(filename)
        img = Image.open(filename)
        mat = np.asarray(img, dtype="int")
        mat = mat.transpose(2, 0, 1)
        temp   = copy.copy(mat[0])
        mat[0] = mat[2]
        mat[2] = temp

        data = ""
        count = 0
        print mat.shape
        for m in mat:
            for r in m:
                count += 1
                data += ",".join(str(i) for i in r)
                data += ","
        data = data[:-1]
        #data = data.split(',')
        # print "len(data): {0}".format(len(data.split(',')))

        top10.append(data)

        #top10.append(('/static/' + path + '?', values))
        #top10 += str(label) + "->" + str(path) + "->" + str(values)
        #top10 += ","

        break

    return top10

def get_top10( retrieval, filename ):
    images = []
    img = Image.open(filename)

    x_size, y_size = img.size
    if x_size == 235:
        img = img.resize((47, 55))
    elif x_size == 47:
        pass
    else:
        s_xy = x_size / 4
        e_xy = x_size / 4 + x_size / 2
        box  = (s_xy, s_xy, e_xy, e_xy)
        img = img.crop(box)
        img = img.resize((47, 55))

    images.append(img)
    a, result = retrieval.get_result(images)
    result = result[0]
    top10 = []

    for label, path, values in result:
        top10.append(('/static/youtube_our/' + path + '?', values))


    return top10

#params_file = '/home/dujunyi/work/faceretrieval/params_ourface.pkl'
#train_data_folder = '/home/dujunyi/work/faceretrieval/data/youtube_ourface_train_deepid/'
#retrieval = Retrieval(params_file, train_data_folder, batch_size=1)

if __name__ == '__main__':

    result = get_top10(retrieval, "./static/data/face.jpg")
    print result

