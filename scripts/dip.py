import sys
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

from scipy.ndimage.interpolation import *
from scipy.optimize import minimize

import field

#dipole = lambda x, y, z, mx, my, mz, dx, dy, dz: mz/(((x - dx)**2 + (y - dy)**2 + (z-dz)**2)**1.5)
dipole = lambda x, y, z, mx, my, mz, dx, dy, dz: 3.0*((x - dx)*mx + (y - dy)*my + (z-dz)*mz)/(((x - dx)**2 + (y - dy)**2 + (z-dz)**2)**2.5) + mz/(((x - dx)**2 + (y - dy)**2 + (z-dz)**2)**1.5)

def _full_error(params, *data):
    B0, B1x, B1y, B2x, B2y, mz, dx, dy, dz = params
    coord, Bz = data
    X, Y = coord
    
    z = 0
    err = 0.0
    
    lenX = len(Bz[0])
    lenY = len(Bz)

    for j in range(lenY):
        for i in range(lenX):
            x = X[j][i]
            y = Y[j][i]
            err += (Bz[j][i] - (B0 + B1x*x + B1y*y + B2x*x*x + B2y*y*y + dipole(x, y, z, 0, 0, mz, dx, dy, dz)))**2
    err = err/lenX/lenY
    return err

def dipoleDetector(Bz, threshold = 10):
    grad = np.gradient(Bz)
    return np.abs(grad[0])*np.abs(grad[1])

def main(argv):
    X, Y, Bx, By, Bz, vol, steps = field.readFile(argv[0])
    x_start, y_start, z_start, width, length, height = vol
    x_step, y_step, z_step = steps
    
    lenX = len(Bz[0])
    lenY = len(Bz)
    Bz = np.array(Bz)

    x0 = [30.0, 0.0, 0.0, 0.0, 0.0, 100000, x_start + width/2.0, y_start + length/2.0, -10.0]

    #res = minimize(full_error, x0, ((X,Y), Uhx_0, Uhx_1, Uhx_2, Uhy_0, Uhy_1, Uhy_2), 'Nelder-Mead', tol=1e-6, bounds = bounds)
    res = minimize(_full_error, x0, ((X,Y), Bz),'Powell', tol=1e-6)
    
    print(res)

    B0, B1x, B1y, B2x, B2y, mz, dx, dy, dz = res.x
    Mres = np.array([[B0 + B1x*X[j][i] + B1y*Y[j][i]+ B2x*X[j][i]*X[j][i] + B2y*Y[j][i]*Y[j][i] + dipole(X[j][i], Y[j][i], 0, 0, 0, mz, dx, dy, dz) for i in range(lenX)] for j in range(lenY)])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Bz, rstride=1, cstride=1, cmap=cm.gist_rainbow)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Mres, rstride=1, cstride=1, cmap=cm.gist_rainbow)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Bz-Mres, rstride=1, cstride=1, cmap=cm.gist_rainbow)
    plt.show()
    
if __name__ == "__main__":
    main(sys.argv[1:])
