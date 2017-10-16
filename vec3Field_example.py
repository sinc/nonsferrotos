#!/usr/bin/env pytnon
#
import sys
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import src.vec3Field as v3f
def main(argv):
    data = v3f.readFile(argv[0] if len(argv)>0 else ".\\data\\09c2г_зачищ_top.txt")
    print(data)
    #Bz = cutSlow(data)
    #print(np.std(Bz))
    grad = np.gradient(data.Bz)
    res= np.abs(grad[0])*np.abs(grad[1])
    #lenX = len(Bz[0])
    #lenY = len(Bz)
    #wind = np.array([[(np.sin(np.pi * i/(lenX - 1))**2)*(np.sin(np.pi * j/(lenY - 1))**2) for i in range(lenX)] for j in range(lenY)])
    #res = np.log10(np.abs(fftshift(fft2(wind*Bz))))

    #X, Y = np.meshgrid(np.arange(len(res[0])), np.arange(len(res)))

    fig = plt.figure('Bz')
    ax = fig.gca(projection='3d')
    ax.plot_surface(data.X, data.Y, data.Bz, rstride=1, cstride=1, cmap=cm.gist_rainbow)
        
    fig = plt.figure('doubleGrad detector')
    ax = fig.gca(projection='3d')
    ax.plot_surface(data.X, data.Y, res, rstride=1, cstride=1, cmap=cm.gist_rainbow)

    plt.show()
    
if __name__ == "__main__":
    main(sys.argv[1:])

