import sys
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

sys.path.append('..\\..\\')

import nonsferrotos.src.vec3Field as v3f
import nonsferrotos.src.recMagnetCutter as rMcut

def main(argv):
    data = v3f.readFile(argv[0] if len(argv)>0 else "..\\data\\09g2s_strike.txt")
    print(data)
    
    fig = plt.figure('source')
    ax = fig.gca(projection='3d')
    ax.plot_surface(data.X, data.Y, data.Bz, rstride=1, cstride=1, cmap=cm.gist_rainbow)

    data_cuted = rMcut.magnetCutter(data,[0.0]*6,[[-0.1,0.1]*6])

    fig = plt.figure('source')
    ax = fig.gca(projection='3d')
    ax.plot_surface(data_cuted.X, data_cuted.Y, data_cuted.Bz, rstride=1, cstride=1, cmap=cm.gist_rainbow)
    
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])