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
#import nonsferrotos.src.detectors as dtc
import nonsferrotos.src.detectorAnalyse as dAn

def main(argv):
    data = v3f.readFile(argv[0] if len(argv)>0 else "..\\data\\09g2s_strike.txt")
    print(data)
    
    fig = plt.figure('source')
    ax = fig.gca(projection='3d')
    ax.plot_surface(data.X, data.Y, data.Bz, rstride=1, cstride=1, cmap=cm.gist_rainbow)
    
    data = ctrs.autoRecRegionCutter(data)
    
    fig = plt.figure('cutted')
    ax = fig.gca(projection='3d')
    ax.plot_surface(data.X, data.Y, data.Bz, rstride=1, cstride=1, cmap=cm.gist_rainbow)

    dataPic = v3f.magnToPicture(data.Bz)
    
    resDet,points = dAn.detectorAnalyseData(data,'crack')
    points = points[:10]
    detPic =v3f.magnToPicture(resDet.Bz)

    fig = plt.figure("detector")
    plt.imshow(detPic, cmap=plt.cm.spectral)
    plt.plot(list(zip(*points))[0],list(zip(*points))[1],'wo')
    #plt.plot(list(zip(*maxes2))[1],list(zip(*maxes2))[0],'ro')
    plt.colorbar()
    plt.show()

    fig = plt.figure("data")
    plt.imshow(dataPic, cmap=plt.cm.spectral)
    plt.plot(list(zip(*points))[0],list(zip(*points))[1],'wo')
    #plt.plot(list(zip(*maxes2))[1],list(zip(*maxes2))[0],'ro')
    plt.colorbar()
    plt.show()


    
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
