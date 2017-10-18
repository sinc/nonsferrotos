#!/usr/bin/env pytnon
#
import sys
import numpy as np
from numpy.fft import *

from scipy import arange

#for comparing volumes and steps massives in one string
def compareArrays(m1,m2):
    if(len(m1)!=len(m2)):
        return False
    else:
        for it in range(len(m1)):
            if(m1[it]!=m2[it]):
                return False
        return True
class vec3Field:
    u'Class that contains data of magnetogramm'
    def __init__(self,X,Y,Bx,By,Bz,vol,steps):
        """the class implements magnetic field data with 
       coordinate 2D meshgrid (X,Y);
       field components (Bx,By,Bz);
       vol [x0,y0,z0,Dx,Dy,Dz], where (x0,y0,z0) - start point coordinates, Dx,Dy,Dz - accordiance height, lenght and whidth;
       steps [dx,dy,dz] - grid step massive  """    
        self.X=X
        self.Y=Y
        self.Bx=Bx
        self.By=By
        self.Bz=Bz
        if(len(Bx)==0):
            self.Bx = np.zeros((len(X),len(X[0])))
        if(len(By)==0):
            self.By = np.zeros((len(X),len(X[0])))
        if(len(Bz)==0):
            self.Bz = np.zeros((len(X),len(X[0])))
        self.vol= vol
        self.steps = steps
    def __str__(self):
        return 'initial point:' + str(self.vol[:3])+'\n size of the data space:' +str(self.vol[3:])+'\n steps of grid:'+str(self.steps)
        #alternative simplest verison: return str(vol)+'\t'+str(steps)

    def __add__(v1,v2):
        if(compareArrays(v1.vol,v2.vol) and compareArrays(v1.steps,v2.steps)):
            return vec3Field(v1.X,v1.Y,v1.Bx+v2.Bx,v1.By+v2.By,v1.Bz+v2.Bz,v1.vol,v1.steps)
        else:
            print('vec3Fields have not same form')
            return None

    def __sub__(v1,v2):
        if(compareArrays(v1.vol,v2.vol) and compareArrays(v1.steps,v2.steps)):
            return vec3Field(v1.X,v1.Y,v1.Bx-v2.Bx,v1.By-v2.By,v1.Bz-v2.Bz,v1.vol,v1.steps)
        else:
            print('vec3Fields have not same form')
            return None

def readFile(fileName):
    lines = [line.strip() for line in open(fileName, 'r')]
    s = lines[0].split('\t')
    x_start = float(s[0].replace(',', '.'))
    y_start = float(s[1].replace(',', '.'))
    z_start = float(s[2].replace(',', '.'))
    width = float(s[3].replace(',', '.'))
    length = float(s[4].replace(',', '.'))
    height = float(s[5].replace(',', '.'))
    s = lines[1].split('\t')
    x_step = float(s[0].replace(',', '.'))
    y_step = float(s[1].replace(',', '.'))
    z_step = float(s[2].replace(',', '.'))
    x_steps = int(round(width/x_step) + 1)
    y_steps = int(round(length/y_step) + 1)
    X = [[x_start + i*x_step for i in range(x_steps)] for j in range(y_steps)]
    tmp = [y_start + i*y_step for i in range(y_steps)]
    Y = [[i]*x_steps for i in tmp]
    tan1 = []
    tan2 = []
    norm = []
    for y in range(y_steps):
        t1 = []
        t2 = []
        n0 = []
        for x in range(x_steps):
            l = lines[y*x_steps + x + 2].split('\t')
            t1.append(float(l[3].replace(',', '.')))
            t2.append(float(l[4].replace(',', '.')))
            n0.append(float(l[5].replace(',', '.')))
        tan1.append(t1)
        tan2.append(t2)
        norm.append(n0)
    return vec3Field(X, Y, tan1, tan2, norm, (x_start, y_start, z_start, width, length, height), (x_step, y_step, z_step))
def saveFile(dataToSave, fileName):
    outFile = open(fileName, 'w')
    for dat in dataToSave.vol[:-1]:
        outFile.write(str(dat).replace('.', ',') + '\t')
    outFile.write(str(dataToSave.vol[-1]).replace('.', ',') + '\n')
    outFile.write((str(dataToSave.steps[0]) + '\t' + str(dataToSave.steps[1]) + '\t' + str(dataToSave.steps[2]) + '\n').replace('.', ','))
    x_steps = len(dataToSave.x[0])
    y_steps = len(dataToSave.x)
    for j in range(y_steps):
        for i in range(x_steps):
            outFile.write((str(dataToSave.x[j][i]) + '\t' + str(dataToSave.y[j][i])+ '\t' + str(dataToSave.vol[2]) + '\t' + str(dataToSave.Bx[j][i])+ '\t' + str(dataToSave.By[j][i])+ '\t' + str(dataToSave.Bz[j][i]) + '\n').replace('.', ','))
    outFile.close()


def precShift(data, lx, ly):
    """shift magnetogram by fft """
    lenX = len(data.Bx[0])
    lenY = len(data.Bx)
    BxSpec = fft2(Bx)
    BySpec = fft2(By)
    BzSpec = fft2(Bz)
    L = ifftshift([[np.exp(-1j*(2.0*np.pi*(i/lenX-0.5)*lx+2.0*np.pi*(j/lenY-0.5)*ly)) for i in np.arange(lenX)] for j in np.arange(lenY)])
    return vec3Field(data.X,data.Y,ifft2(BxSpec*L),ifft2(BySpec*L),ifft2(BzSpec*L),data.vol,data.steps)


def nearCoordinates(vol,Xold,Yold,steps,point):
    """find indicies from one coordinate grid to another coordinate grid with same steps"""
    xind = int(round( (Xold[point[1]][point[0]]-vol[0]) / steps[0]))
    yind = int(round( (Yold[point[1]][point[0]]-vol[1]) / steps[1]))
    #print(vol,Xold[0][0],Yold[0][0],steps,point)
    return [xind if xind>=0 and xind < vol[3]/steps[0] else np.nan,
            yind if yind>=0 and yind < vol[4]/steps[1] else np.nan]
