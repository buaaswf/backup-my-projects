import SocketServer
from SocketServer import StreamRequestHandler as SRH

from scipy.io import loadmat
import numpy as np
import cPickle as pickle
import math
from config import VERIFY_RATIO_HOST, VERIFY_RATIO_PORT
from sklearn import svm
from sklearn.externals import joblib

host = '192.168.1.1'
port = VERIFY_RATIO_PORT
addr = (host, port)

RECV_SIZE = 1024
import cPickle as pickle

class Servers(SRH):
    def handle(self):
        print 'Get connection from ',self.client_address
        #  self.wfile.write('Connection %s:%s at %s succeed!' % (host, port, ctime()))
        while True:
            mat_file = self.request.recv(RECV_SIZE)
            print "recv mat_file:", mat_file

            result = "Running error."
            try:
                result = cosine_verify(mat_file)
            except Exception, e:
                print str(e)

            self.request.sendall(str(result))

            break

#  gender_model_file = "/home/s.li/cyh/works/cnn_lbp/model/face_verify/age_0/svm_clt.pkl"
#  gender_clf = joblib.load(gender_model_file)
#  f = open("/home/s.li/cyh/works/cnn_lbp/model/face_verify/age_0/svm_data_mean.pkl")
#  gender_mean = pickle.load(f)['mean']
#  gender_mean = gender_mean.reshape(gender_mean.shape[0],1).T

def gender_classify(fea):
    try:
        fea = fea.reshape(fea.shape[0],1).T
        fea -= gender_mean

        pre = gender_clf.predict(fea)
    except Exception, e:
        print "gender_classify:", e
    return pre[0]

def cosine_verify(mat_file):
    mat = loadmat(mat_file)
    data  = np.asarray(mat['data'], dtype='float')

    cosine_distance = SingleCosine(data[0], data[1])
    print cosine_distance

    return cosine_distance

    if cosine_distance > 0.5:
        return "Same Person. Cosine distance: %f"%cosine_distance

    return "Different Person. Cosine distance: %f"%cosine_distance

def SingleCosine(a, b):
    dot = [0 for i in range(len(a))]
    for i in range(len(a)):
        dot[i] = a[i] * b[i]
    sum_a = math.sqrt(sum([i ** 2 for i in a]))
    sum_b = math.sqrt(sum([i ** 2 for i in b]))
    res = sum(dot) / (sum_a * sum_b)

    return res

if __name__ == "__main__":
    print 'Server is running....'
    server = SocketServer.ThreadingTCPServer(addr,Servers)
    server.serve_forever()

