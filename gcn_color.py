from PIL import Image
import numpy as np
import math
def global_contrast_normalization(filename, s, lmda, epsilon):
    X = np.array(Image.open(filename))

    X_prime=X
    r,c,u=X.shape
    contrast =0
    su=0
    sum_x=0

    for i in range(r):
        for j in range(c):
            for k in range(u):

                sum_x=sum_x+X[i][j][k]
    X_average=float(sum_x)/(r*c*u)

    for i in range(r):
        for j in range(c):
            for k in range(u):

                su=su+((X[i][j][k])-X_average)**2
    contrast=np.sqrt(lmda+(float(su)/(r*c*u)))


    for i in range(r):
        for j in range(c):
            for k in range(u):

                X_prime[i][j][k] = s * (X[i][j][k] - X_average) / max(epsilon, contrast)
    Image.fromarray(X_prime).save("result.jpg")
global_contrast_normalization("gen_n02025239_58.JPEG", 1, 10, 0.000000001)