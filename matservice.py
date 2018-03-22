import SocketServer
from SocketServer import StreamRequestHandler as SRH
from time import ctime
from time import sleep
import numpy as np
import featureservice
import sys
import Image

host = '0.0.0.0'
port = 3201
addr = (host, port)
net, transformer = featureservice.load_model()
#net,transformer=featureservice.load_models("models/scene/idcard/deepid_flip_train_iter_20000.caffemodel",
#"models/scene/idcard/deploy.prototxt","models/scene/idcard/idface.npy",[256,3,55,55])
#net,transformer=featureservice.load_models("models/mfm/model/DeepFace_set003_net_iter.caffemodel",
#"models/mfm/proto/DeepFace_set003.prototxt","models/mfm/data/alignCASIA-WebFace_mean_2.npy",[20,1,128,128])
RECV_SIZE = 1024

class Servers(SRH):
    def handle(self):
        print 'Got connection from ',self.client_address
       # self.wfile.write('Connection %s:%s at %s succeed!' % (host, port, ctime()))
        #while True:
        try:
            rev_data = self.request.recv(RECV_SIZE)
        #self.wfile.write('Recv: %s' % rev_data)
        except Exception as e:
            print str(e)
        dirs=rev_data.split(";")
        featureservice.image2mat(dirs[0],dirs[1],"/home/s.li/web-g206/verify_images/pair.mat", net, transformer)
        self.request.sendall("/home/s.li/web-g206/verify_images/pair.mat")
if __name__ == "__main__":
    print 'Server is running....'
    server = SocketServer.ThreadingTCPServer(addr,Servers)
    server.serve_forever()

