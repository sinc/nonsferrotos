import numpy as np
import scipy.optimize as opt

from . import vec3Field as v3f
from . import cutters as ctrs

#
dipoleLambdaZ = lambda x, y, z, mx, my, mz, dx, dy, dz: [3.0*((x - dx)*mx + (y - dy)*my + (z-dz)*mz)*(x-dx)/(((x - dx)**2 + (y - dy)**2 + (z-dz)**2)**2.5) - mx/(((x - dx)**2 + (y - dy)**2 + (z-dz)**2)**1.5),
                                                         3.0*((x - dx)*mx + (y - dy)*my + (z-dz)*mz)*(y-dy)/(((x - dx)**2 + (y - dy)**2 + (z-dz)**2)**2.5) - my/(((x - dx)**2 + (y - dy)**2 + (z-dz)**2)**1.5),
                                                         3.0*((x - dx)*mx + (y - dy)*my + (z-dz)*mz)*(z-dz)/(((x - dx)**2 + (y - dy)**2 + (z-dz)**2)**2.5) - mz/(((x - dx)**2 + (y - dy)**2 + (z-dz)**2)**1.5)]
def _fullDipole_error(params, *data):
    B0, B1x, B1y, B2x, B2y,mx,my, mz,dx, dy, dz = params
    coord, Bz,  = data
    X, Y = coord
    
    z = 0
    err = 0.0
    
    lenX = len(Bz[0])
    lenY = len(Bz)

    for j in range(lenY):
        for i in range(lenX):
            x = X[j][i]
            y = Y[j][i]
            field = dipoleLambdaZ(x, y, z, mx, my, mz, dx, dy, dz)
            err += (Bz[j][i] - (B0 + B1x*x + B1y*y+B2x*x*x+B2y*y*y+field[2]))**2
    err = err/lenX/lenY
    #print (err,(dx,dy,dz),(mx,my,mz))
    return err
def oneDipoleCutter(field,point):    
    point = [point[1],point[0]]
    print('pointToAnalyse:',point)
    dx = field.vol[0]+field.vol[3]/2.0
    dy = field.vol[1]+field.vol[4]/2.0
    x0 = [0, 0, 0, 0, 0, 0, 0, 0,dx,dy,-16.0]
    test = opt.minimize(_fullDipole_error, x0, ((field.X,field.Y), field.Bz),'Powell', tol=1e-3)
    B0, B1x, B1y, B2x, B2y, mx, my, mz, dx, dy, dz = test["x"]
    return [mx, my, mz, dx, dy, dz]
def dipoleCutter(field,points,winHalfSize):
    dipoleData=[oneDipoleCutter(ctrs.regionCutter(field,points[0],winHalfSize),points[0])]
    dipoleField = np.array([[dipoleLambdaZ(field.X[j][i], field.Y[j][i], 0, *dipoleData[-1])[2] for i in range(len(field.X[0]))] for j in range(len(field.X))])
    for point in points[1:]: 
        dipoleData.append(oneDipoleCutter(ctrs.regionCutter(field,point,winHalfSize),point)
        )
        dipoleField += np.array([[dipoleLambdaZ(field.X[j][i], field.Y[j][i], 0, *dipoleData[-1])[2] for i in range(len(field.X[0]))] for j in range(len(field.X))])
        #dipoleField += np.array([[dipoleLambdaZ(X[j][i], Y[j][i], 0, mx, my, mz, dx, dy, dz)[2] for i in range(len(X[0]))] for j in range(len(X))])
    return dipoleData, v3f.vec3Field(field.X,field.Y,[],[],dipoleField,field.vol,field.steps)