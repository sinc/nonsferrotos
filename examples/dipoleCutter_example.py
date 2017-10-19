#!/usr/bin/env pytnon
#
import sys
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from skimage.feature import peak_local_max

sys.path.append('..\\..\\')

import nonsferrotos.src.vec3Field as v3f
import nonsferrotos.src.detectors as dtc
import nonsferrotos.src.dipoleCutter as dpctr

def main(argv):
    data = v3f.readFile(argv[0] if len(argv)>0 else "..\\data\\09g2s_strike.txt")
    print(data)
    winsize = 4
    winHalfSize = 8
    countOfDipolesToAnalyse = 1
    #calc crack-detector
    resCrack = dtc.gradDetector(data,winsize,'crack')
    #find maxes of crack-detector
    maxes = peak_local_max(np.array(resCrack.Bz),min_distance = 3 , indices =True)
    #sort maxes in 
    maxes = sorted(maxes,key = lambda item: resCrack.Bz[item[0]][item[1]],reverse = True)
    #find inicies of maxes in source data
    maxes = [v3f.nearIndicies(data.vol,resCrack.X,resCrack.Y,data.steps,[point[1],point[0]]) for point in maxes]
    print(maxes)
    #
    dipoles, dipoleField = dpctr.dipoleCutter(data,maxes[:countOfDipolesToAnalyse],winHalfSize)
    fig = plt.figure('source')
    ax = fig.gca(projection='3d')
    ax.plot_surface(data.X, data.Y, data.Bz, rstride=1, cstride=1, cmap=cm.gist_rainbow)
    
    fig = plt.figure('dipole Field')
    ax = fig.gca(projection='3d')
    ax.plot_surface(dipoleField.X, dipoleField.Y, dipoleField.Bz, rstride=1, cstride=1, cmap=cm.gist_rainbow)

    fig = plt.figure('substracted field')
    ax = fig.gca(projection='3d')
    ax.plot_surface(dipoleField.X, dipoleField.Y, (dipoleField-data).Bz, rstride=1, cstride=1, cmap=cm.gist_rainbow)

    plt.show()
    
if __name__ == "__main__":
    main(sys.argv[1:])


