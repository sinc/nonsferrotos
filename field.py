#!/usr/bin/env pytnon
#
import sys
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from scipy import arange
from numpy.fft import *

from scipy.optimize import minimize

def readFile(fileName):
    lines = [line.strip() for line in open(fileName, 'r')]
    s = lines[0].split('\t')
    x_start = float(s[0].replace(',', '.'))
    y_start = float(s[1].replace(',', '.'))
    z_start = float(s[2].replace(',', '.'))
    width = float(s[3].replace(',', '.'))
    length = float(s[4].replace(',', '.'))
    height = float(s[5].replace(',', '.'))
    s = lines[1].split('\t')
    x_step = float(s[0].replace(',', '.'))
    y_step = float(s[1].replace(',', '.'))
    z_step = float(s[2].replace(',', '.'))
    x_steps = int(round(width/x_step) + 1)
    y_steps = int(round(length/y_step) + 1)
    X = [[x_start + i*x_step for i in range(x_steps)] for j in range(y_steps)]
    tmp = [y_start + i*y_step for i in range(y_steps)]
    Y = [[i]*x_steps for i in tmp]
    tan1 = []
    tan2 = []
    norm = []
    for y in range(y_steps):
        t1 = []
        t2 = []
        n0 = []
        for x in range(x_steps):
            l = lines[y*x_steps + x + 2].split('\t')
            t1.append(float(l[3].replace(',', '.')))
            t2.append(float(l[4].replace(',', '.')))
            n0.append(float(l[5].replace(',', '.')))
        tan1.append(t1)
        tan2.append(t2)
        norm.append(n0)
    return X, Y, tan1, tan2, norm, (x_start, y_start, z_start, width, length, height), (x_step, y_step, z_step)

def saveFile(fileName, vol, steps, x, y, tan1, tan2, norm):
    outFile = open(fileName, 'w')
    for dat in vol[:-1]:
        outFile.write(str(dat).replace('.', ',') + '\t')
    outFile.write(str(vol[-1]).replace('.', ',') + '\n')
    outFile.write((str(steps[0]) + '\t' + str(steps[1]) + '\t' + str(steps[2]) + '\n').replace('.', ','))
    x_steps = len(x[0])
    y_steps = len(x)
    for j in range(y_steps):
        for i in range(x_steps):
            outFile.write((str(x[j][i]) + '\t' + str(y[j][i])+ '\t' + str(vol[2]) + '\t' + str(tan1[j][i])+ '\t' + str(tan2[j][i])+ '\t' + str(norm[j][i]) + '\n').replace('.', ','))
    outFile.close()

def precShift(B, lx, ly):
    lenX = len(B[0])
    lenY = len(B)
    BSpec = fft2(B)
    L = ifftshift([[np.exp(-1j*(2.0*np.pi*(i/lenX-0.5)*lx+2.0*np.pi*(j/lenY-0.5)*ly)) for i in np.arange(lenX)] for j in np.arange(lenY)])
    return ifft2(BSpec*L)

def cutSlow(B, X, Y):
    def _full_error(params, *data):
        B0, B1x, B1y, B2x, B2y, B3x, B3y = params
        coord, Bz = data
        X, Y = coord
    
        lenX = len(Bz[0])
        lenY = len(Bz)

        err = 0.0
        for j in range(lenY):
            for i in range(lenX):
                x = X[j][i]
                y = Y[j][i]
                err += (Bz[j][i] - (B0 + B1x*x + B1y*y + B2x*x*x + B2y*y*y+ B3x*x*x*x + B3y*y*y*y))**2
        err = err/lenX/lenY
        return err

    lenX = len(B[0])
    lenY = len(B)
    B = np.array(B)

    x0 = [30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    res = minimize(_full_error, x0, ((X,Y), B), 'Powell', tol=1e-6)
    print(res)
    B0, B1x, B1y, B2x, B2y, B3x, B3y = res.x
    return B - np.array([[B0 + B1x*X[j][i] + B1y*Y[j][i]+ B2x*X[j][i]*X[j][i] + B2y*Y[j][i]*Y[j][i] + B3x*(X[j][i]**3) + B3y*(Y[j][i]**3) for i in range(lenX)] for j in range(lenY)])

"""
#Bz = cutSlow(norm, X, Y)
    
    #lenX = len(Bz[0])
    #lenY = len(Bz)
    #wind = np.array([[(np.sin(np.pi * i/(lenX - 1))**2)*(np.sin(np.pi * j/(lenY - 1))**2) for i in range(lenX)] for j in range(lenY)])
    #res = np.log10(np.abs(fftshift(fft2(wind*Bz))))

    X, Y = np.meshgrid(np.arange(len(res[0])), np.arange(len(res)))

    #levels = MaxNLocator(nbins=1000).tick_values(res.min(), res.max())
    #cmap = plt.get_cmap('hot')
    #norm1 = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    #fig, ax = plt.subplots()
    #im = ax.pcolormesh(X, Y, res, cmap=cmap, norm=norm1)
    #cbar = fig.colorbar(im)
    #fig.tight_layout()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, res, rstride=1, cstride=1, cmap=cm.gist_rainbow)
    plt.show()
"""

def main(argv):
    X, Y, tan1, tan2, norm, vol, steps = readFile(argv[0])
    Bz = cutSlow(norm, X, Y)
    #print(np.std(Bz))
    grad = np.gradient(norm)
    res= np.abs(grad[0])*np.abs(grad[1])
    #lenX = len(Bz[0])
    #lenY = len(Bz)
    #wind = np.array([[(np.sin(np.pi * i/(lenX - 1))**2)*(np.sin(np.pi * j/(lenY - 1))**2) for i in range(lenX)] for j in range(lenY)])
    #res = np.log10(np.abs(fftshift(fft2(wind*Bz))))

    #X, Y = np.meshgrid(np.arange(len(res[0])), np.arange(len(res)))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, norm, rstride=1, cstride=1, cmap=cm.gist_rainbow)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Bz, rstride=1, cstride=1, cmap=cm.gist_rainbow)
        
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, res, rstride=1, cstride=1, cmap=cm.gist_rainbow)

    plt.show()
    
if __name__ == "__main__":
    main(sys.argv[1:])
