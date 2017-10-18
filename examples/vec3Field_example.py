﻿#!/usr/bin/env pytnon
#
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('..\\..\\')
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import vec3Field as v3f
#plot three components of magnetic field from file
def main(argv):
    data = v3f.readFile(argv[0] if len(argv)>0 else "..\\data\\09c2г_зачищ_top.txt")
    print(data)
    
    fig = plt.figure('Bx')
    ax = fig.gca(projection='3d')
    ax.plot_surface(data.X, data.Y, data.Bx, rstride=1, cstride=1, cmap=cm.gist_rainbow)
    
    fig = plt.figure('By')
    ax = fig.gca(projection='3d')
    ax.plot_surface(data.X, data.Y, data.By, rstride=1, cstride=1, cmap=cm.gist_rainbow)

    fig = plt.figure('Bz')
    ax = fig.gca(projection='3d')
    ax.plot_surface(data.X, data.Y, data.Bz, rstride=1, cstride=1, cmap=cm.gist_rainbow)

    plt.show()
    
if __name__ == "__main__":
    main(sys.argv[1:])

