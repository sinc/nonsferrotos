    #!/usr/bin/env pytnon
#
import sys
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

from matplotlib import rc
rc('font',**{'family':'serif'})
rc('text', usetex=True)
rc('text.latex',unicode=True)
rc('text.latex',preamble='\\usepackage[utf8]{inputenc}')
rc('text.latex',preamble='\\usepackage[russian]{babel}')

sys.path.append('..\\..\\')
import nonsferrotos.src.vec3Field as v3f
import nonsferrotos.src.detectorAnalyse as dAn
import nonsferrotos.src.crackFill as crkf


def main(argv):
    data = v3f.readFile(argv[0] if len(argv)>0 else "..\\data\\mera\\mera_with_magnitezation_gap=100um.txt")
    print(data)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=18)
    fig = plt.figure('source')
    ax = fig.gca(projection='3d')
    ax.plot_surface(data.X, data.Y, data.Bz, rstride=1, cstride=1, cmap=cm.gist_rainbow)
    
    resDet,points = dAn.detectorAnalyseData(data,'crack',typeOfResult = 'coords',winSize = 3)
    resDet.Bz = np.abs(resDet.Bz)
    resDet.Bz/= np.max(resDet.Bz)
    points = points[:3]
    maxes = [v3f.indexFromCoord(resDet.vol,resDet.steps,[point[1],point[0]]) for point in points]
    print(maxes)
    maxes = [[point[1],point[0]] for point in maxes]
    print(maxes)
    fig = plt.figure('detector')
    plt.xlabel('$x$, mm', fontsize=18)
    plt.ylabel('$y$, mm', fontsize=18)
    #plt.colorlabel('$Bz, \mu T$', fontsize=18)
    plt.pcolor(resDet.X,resDet.Y,resDet.Bz,cmap=plt.cm.spectral)
    plt.plot(list(zip(*points))[1],list(zip(*points))[0],'wo')
    clb = plt.colorbar()
    clb.set_label('|$D_{x+y}(x,y)|/D_{x+y(m)}$', fontsize=18)
    
    plt.show()
    
    dataCrack = crkf.lineGeneration2(maxes, resDet.Bz)
    fig = plt.figure('line')
    plt.xlabel('$x$, mm', fontsize=18)
    plt.ylabel('$y$, mm', fontsize=18)
    #plt.colorlabel('$Bz, \mu T$', fontsize=18)
    plt.pcolor(resDet.X,resDet.Y,dataCrack,cmap=plt.cm.spectral)
    plt.plot(list(zip(*points))[1],list(zip(*points))[0],'wo')
    
    plt.show()
    print('ok')
if __name__ == "__main__":
    main(sys.argv[1:])

