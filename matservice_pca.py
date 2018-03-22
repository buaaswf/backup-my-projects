import SocketServer
from SocketServer import StreamRequestHandler as SRH
from time import ctime
from time import sleep
import numpy as np
import featureservice
import sys
import Image

host = '0.0.0.0'
port = 3202
addr = (host, port)
net, transformer = featureservice.load_model_1213()

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

        featureservice.image2mat_pca(dirs[0],dirs[0][:-4] + ".mat",
                net,transformer,dirs[1])
        self.request.sendall(dirs[0][:-4]+".mat")
if __name__ == "__main__":
    print 'Server is running....'
    server = SocketServer.ThreadingTCPServer(addr,Servers)
    server.serve_forever()

