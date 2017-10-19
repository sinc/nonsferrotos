import sys
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from numpy.fft import *

from scipy.ndimage.interpolation import *

import field

def laplasDetector(B, sx = 2, sy = 2):
    lenX = len(B[0])
    lenY = len(B)
    # 1  1  1
    # 1 -8  1
    # 1  1  1
    result = []
    for l in range(lenY-2*sy):
        t = []
        for k in range(lenX-2*sx):
            t.append(B[l][k] + B[l][k+sx] + B[l][k+2*sx] + B[l+sy][k] + B[l+sy][k+2*sx] + B[l+2*sy][k] + B[l+2*sy][k+sx] + B[l+2*sy][k+2*sx] - 8*B[l+sy][k+sx])
        result.append(t)
    return result


def main(argv):
    Z0 = 3.5
    
    X, Y, Bx, By, Bz, vol, steps = field.readFile(argv[0])
    x_start, y_start, z_start, width, length, height = vol
    x_step, y_step, z_step = steps
    
    lenX = len(Bz[0])
    lenY = len(Bz)

    #Bz1 = [[0 for i in range(lenX)] for j in range(lenY)]
    #for i in range(len(Bz[0])):
    #    for j in range(len(Bz)):
    #        Bz1[j][i] = Bz[j][i]
    #аппаратная функция
    L = [[np.exp(Z0*((i/lenX-0.5)**2.0+(j/lenY-0.5)**2.0)**0.5) for i in np.arange(lenX)] for j in np.arange(lenY)]
    
    BzSpec = fft2(Bz)
    Mres   = ifft2(2.0*BzSpec*L).real
    
    #lenX = len(Mres[0])
    #lenY = len(Mres)

    X, Y = np.meshgrid(np.arange(lenX), np.arange(lenY))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Mres, rstride=1, cstride=1, cmap=cm.gist_rainbow)
    plt.show()
    
if __name__ == "__main__":
    main(sys.argv[1:])
