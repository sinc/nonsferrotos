#!/usr/bin/env pytnon
#
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('..\\..\\')
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

import nonsferrotos.src.vec3Field as v3f
import nonsferrotos.src.detectors as dtc

def main(argv):
    data = v3f.readFile(argv[0] if len(argv)>0 else "..\\data\\09c2г_зачищ_top.txt")
    print(data)
    winsize = 4

    resDipole = dtc.gradDetector(data,winsize,'dipole')
    resDGrad = dtc.gradDetector(data,winsize,'doubleGrad')
    resCrack = dtc.gradDetector(data,winsize,'crack')

    fig = plt.figure('resDipole')
    ax = fig.gca(projection='3d')
    ax.plot_surface(resDipole.X, resDipole.Y, resDipole.Bz, rstride=1, cstride=1, cmap=cm.gist_rainbow)
    
    fig = plt.figure('resDGrad')
    ax = fig.gca(projection='3d')
    ax.plot_surface(resDGrad.X, resDGrad.Y, resDGrad.Bz, rstride=1, cstride=1, cmap=cm.gist_rainbow)    

    fig = plt.figure('resCrack')
    ax = fig.gca(projection='3d')
    ax.plot_surface(resCrack.X, resCrack.Y, resCrack.Bz, rstride=1, cstride=1, cmap=cm.gist_rainbow)
    

    plt.show()
    
if __name__ == "__main__":
    main(sys.argv[1:])
