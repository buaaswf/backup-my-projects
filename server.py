import SocketServer
from SocketServer import StreamRequestHandler as SRH
from time import ctime
from time import sleep
import numpy as np

import sys
import Image

host = '10.2.15.11'
port = 9797
addr = (host, port)

RECV_SIZE = 1024

class Servers(SRH):
    def handle(self):
        print 'Got connection from ',self.client_address
        self.wfile.write('Connection %s:%s at %s succeed!' % (host, port, ctime()))
        data = ""
        while True:
            rev_data = self.request.recv(RECV_SIZE)
            
            self.wfile.write('Recv: %s' % rev_data)

            self.request.sendall(rev_data)

if __name__ == "__main__":
    print 'Server is running....'
    server = SocketServer.ThreadingTCPServer(addr,Servers)
    server.serve_forever()

