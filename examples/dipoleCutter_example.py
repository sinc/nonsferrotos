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
import nonsferrotos.src.cutters as ctrs
import nonsferrotos.src.dipoleCutter as dpctr
import nonsferrotos.src.detectorAnalyse as dAn
def main(argv):
    data = v3f.readFile(argv[0] if len(argv)>0 else "..\\data\\09g2s_strike.txt")
    print(data)
    winsize = 4
    winHalfSize = 8
    countOfDipolesToAnalyse = 2
    
    #auto cut area under the sample
    data = ctrs.autoRecRegionCutter(data)
    #find maxes of dipole detector
    resDet, maxes = dAn.detectorAnalyseData(data,'crack',typeOfResult = 'indicies')
    resDet, maxesCoords = dAn.detectorAnalyseData(data,'crack',typeOfResult = 'coords')
    print(maxes)
    #dipole Cutter
    dipoles, dipoleField = dpctr.dipoleCutter(data,maxes[:countOfDipolesToAnalyse],winHalfSize)

    fig = plt.figure('cuted source')
    plt.xlabel('$x$, mm', fontsize=18)
    plt.ylabel('$y$, mm', fontsize=18)
    plt.pcolor(data.X, data.Y, data.Bz, cmap=plt.cm.spectral)
    clb = plt.colorbar()
    clb.set_label('$B_z, \mu T$', fontsize=18)
    
    fig = plt.figure('detector with maxes')
    plt.xlabel('$x$, mm', fontsize=18)
    plt.ylabel('$y$, mm', fontsize=18)
    plt.pcolor(resDet.X, resDet.Y, (resDet).Bz,cmap=plt.cm.spectral)
    plt.plot(list(zip(*maxesCoords[countOfDipolesToAnalyse:]))[1],list(zip(*maxesCoords[countOfDipolesToAnalyse:]))[0],'wx')
    plt.plot(list(zip(*maxesCoords[:countOfDipolesToAnalyse]))[1],list(zip(*maxesCoords[:countOfDipolesToAnalyse]))[0],'rx')
    clb = plt.colorbar()
    clb.set_label('$B_z, \mu T$', fontsize=18)

    fig = plt.figure('dipole Field')
    plt.xlabel('$x$, mm', fontsize=18)
    plt.ylabel('$y$, mm', fontsize=18)
    plt.pcolor(dipoleField.X, dipoleField.Y, dipoleField.Bz,cmap=plt.cm.spectral)
    clb = plt.colorbar()
    clb.set_label('$B_z, \mu T$', fontsize=18)
  
    fig = plt.figure('substracted field')
    plt.xlabel('$x$, mm', fontsize=18)
    plt.ylabel('$y$, mm', fontsize=18)
    plt.pcolor(dipoleField.X, dipoleField.Y, (dipoleField-data).Bz,cmap=plt.cm.spectral)
    clb = plt.colorbar()
    clb.set_label('$B_z, \mu T$', fontsize=18)

    plt.show()
    print('ok')
if __name__ == "__main__":
    main(sys.argv[1:])


