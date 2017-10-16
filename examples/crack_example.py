import sys
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import vec3Field as v3f
import models 

xsize = 100 
ysize = 100
height = 10

X,Y = np.meshgrid(range(xsize),range(ysize)) #
vol  = [0,0,height,xsize,ysize,0]
steps = [1,1,1]
data = v3f.vec3Field(X,Y,None,None,None,vol,steps)
coord,magnetization = models.randomCrackExampleLinearModel(vol)
for i in range(len(X)):
    for j in range(len(X[0])):
        data.Bx[i][j],data.By[i][j],data.Bz[i][j] = models.crack(X[i][j],Y[i][j],0,coord,magnetization)
fig = plt.figure('Bz')
ax = fig.gca(projection='3d')
ax.plot_surface(data.X, data.Y, data.Bz, rstride=1, cstride=1, cmap=cm.gist_rainbow)
        
plt.show()