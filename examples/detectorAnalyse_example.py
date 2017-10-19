#!/usr/bin/env pytnon
#
import sys
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

sys.path.append('..\\..\\')
import nonsferrotos.src.vec3Field as v3f
import nonsferrotos.src.cutters as ctrs

def main(argv):
    data = v3f.readFile(argv[0] if len(argv)>0 else "..\\data\\09g2s_strike.txt")
    print(data)
    
    fig = plt.figure('source')
    ax = fig.gca(projection='3d')
    ax.plot_surface(data.X, data.Y, data.Bz, rstride=1, cstride=1, cmap=cm.gist_rainbow)
    
    fig = plt.figure('cutted')
    ax = fig.gca(projection='3d')
    ax.plot_surface(data2.X, data2.Y, data2.Bz, rstride=1, cstride=1, cmap=cm.gist_rainbow)
    

    plt.show()
    
if __name__ == "__main__":
    main(sys.argv[1:])
