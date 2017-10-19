import sys
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from numpy.fft import *

import field

def main(argv):
    #параметры регуляризации
    p = 1.01
    alpha = 0.01
    Z0 = 3.0
    
    X, Y, Bx, By, Bz, vol, steps = field.readFile(argv[0])
    x_start, y_start, z_start, width, length, height = vol
    x_step, y_step, z_step = steps
    
    lenX = len(Bz[0])
    lenY = len(Bz)
    
    Bz1 = [[0 for i in range(lenX)] for j in range(lenY)]
    for i in range(len(Bz[0])):
        for j in range(len(Bz)):
            Bz1[j][i] = Bz[j][i]
    
    #сетка для регуляризатора и графиков
    L = np.array([[((i/lenX-0.5)**2.0)**p+((j/lenY-0.5)**2.0)**p for i in np.arange(lenX)] for j in np.arange(lenY)])
    #временная сетка для системной функции
    #X_grid = np.arange(-width/2.0, width/2.0+x_step, x_step)
    #Y_grid = np.arange(-length/2.0, length/2.0+y_step, y_step)
    X_grid = np.linspace(-0.5, 0.5, lenX)
    Y_grid = np.linspace(-0.5, 0.5, lenY)

    BzSpec = fft2(Bz1)
    #K = [[(x_**2 + y_**2 - 2.0*Z0**2)/((x_**2 + y_**2 + Z0**2)**2.5) for x_ in X_grid] for y_ in Y_grid]
    K = [[Z0/((x_**2 + y_**2 + Z0**2)**1.5) for x_ in X_grid] for y_ in Y_grid]
    KSpec = fft2(K)
    MSpec = BzSpec*KSpec / (KSpec*np.conj(KSpec) + alpha*L)
    Bzres = ifft2(MSpec*KSpec)
    M = ifft2(MSpec)
    #res = ((Bz1-Bzres.real)**2).ravel()
    #rms = sum(res)/len(res)
    #print(Z0, rms)
    res = Bz1-Bzres.real
    X, Y = np.meshgrid(np.arange(lenX), np.arange(lenY))
    #ret1 = [rr[:-1] for rr in res[:-1]]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, M.real, rstride=1, cstride=1, cmap=cm.gist_rainbow)
    plt.show()
    
if __name__ == "__main__":
    main(sys.argv[1:])
