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
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=18)
    
    fig = plt.figure()
    plt.xlabel('$x$, mm', fontsize=18)
    plt.ylabel('$y$, mm', fontsize=18)
    #plt.colorlabel('$Bz, \mu T$', fontsize=18)
    plt.pcolor(data.X,data.Y,data.Bz,cmap=plt.cm.spectral)
   
    #plt.plot(list(zip(*maxes2))[1],list(zip(*maxes2))[0],'ro')
    clb = plt.colorbar()
    clb.set_label('$B_z$, A/m', fontsize=18)
    #plt.savefig('source')
    plt.show()
    
    data = ctrs.autoRecRegionCutter(data)
    
    fig = plt.figure()
    plt.xlabel('$x$, mm', fontsize=18)
    plt.ylabel('$y$, mm', fontsize=18)
    #plt.colorlabel('$Bz, \mu T$', fontsize=18)
    plt.pcolor(data.X,data.Y,data.Bz,cmap=plt.cm.spectral)
   
    #plt.plot(list(zip(*maxes2))[1],list(zip(*maxes2))[0],'ro')
    clb = plt.colorbar()
    clb.set_label('$B_z$, A/m', fontsize=18)
    #plt.savefig('cutedSource')
    plt.show()

    #dataPic = v3f.magnToPicture(data.Bz)
    
    resDet,points = dAn.detectorAnalyseData(data,'crack',typeOfResult = 'coords')
    points = points[:5]
    
    fig = plt.figure()
    plt.xlabel('$x$, mm', fontsize=18)
    plt.ylabel('$y$, mm', fontsize=18)
    #plt.colorlabel('$Bz, \mu T$', fontsize=18)
    plt.pcolor(resDet.X,resDet.Y,resDet.Bz,cmap=plt.cm.spectral)
    #plt.plot(list(zip(*maxes2))[1],list(zip(*maxes2))[0],'ro')
    clb = plt.colorbar()
    clb.set_label('detector', fontsize=18)
    #plt.savefig('detector')
    plt.show()

    fig = plt.figure("detectorWithMaxes")    
    plt.xlabel('$x$, mm', fontsize=18)
    plt.ylabel('$y$, mm', fontsize=18)
    #plt.colorlabel('$Bz, \mu T$', fontsize=18)
    plt.pcolor(resDet.X,resDet.Y,resDet.Bz,cmap=plt.cm.spectral)
   
    #plt.plot(list(zip(*maxes2))[1],list(zip(*maxes2))[0],'ro')
    clb = plt.colorbar()
    clb.set_label('detector', fontsize=18)
    plt.plot(list(zip(*points))[1],list(zip(*points))[0],'wo')
    #plt.savefig('sourceWithMax')
    plt.show()
    print('ok')
    
if __name__ == "__main__":
    main(sys.argv[1:])
