import numpy as np
from . import vec3Field as v3f
from scipy.optimize import minimize

from skimage.feature import peak_local_max
#cut square with 2*winHalfSize size region around the point (pointOfCenter)
#if point is near the edges of source data then cutted region includes only existense points
def regionCutter(field,pointOfCenter,winHalfSize):
    print(field)
    startXIndex =pointOfCenter[0]-winHalfSize if pointOfCenter[0]-winHalfSize>=0 else 0 
    startYIndex =pointOfCenter[1]-winHalfSize if pointOfCenter[1]-winHalfSize>=0 else 0 
    stopXIndex =pointOfCenter[0]+winHalfSize if pointOfCenter[0]+winHalfSize-len(field.X[0])<0 else len(field.X[0])-1
    stopYIndex =pointOfCenter[1]+winHalfSize if pointOfCenter[1]+winHalfSize-len(field.X)<0 else len(field.X)-1
    print('bounds of cuted region:',startXIndex,stopXIndex,startYIndex,stopYIndex)
    volNew = [field.X[startYIndex][startXIndex],field.Y[startYIndex][startXIndex],field.vol[2],(stopXIndex-startXIndex)*field.steps[0],(stopYIndex-startYIndex)*field.steps[1],0.0] 
    X1, Y1  = np.meshgrid(
        np.arange(volNew[0],volNew[0]+volNew[3],field.steps[0]), 
        np.arange(volNew[1],volNew[1]+volNew[4],field.steps[1]),
        )
    return v3f.vec3Field(X1,Y1,np.array(field.Bx)[startYIndex:stopYIndex,startXIndex:stopXIndex],
                         np.array(field.By)[startYIndex:stopYIndex,startXIndex:stopXIndex],
                         np.array(field.Bz)[startYIndex:stopYIndex,startXIndex:stopXIndex],volNew,field.steps
                         )

def cutSlowProcesser(B, X, Y):
    def _full_error(params, *data):
        B0, B1x, B1y, B2x, B2y, B3x, B3y = params
        coord, Bz = data
        X, Y = coord
        lenX = len(Bz[0])
        lenY = len(Bz)
        err = 0.0
        for j in range(lenY):
            for i in range(lenX):
                x = X[j][i]
                y = Y[j][i]
                err += (Bz[j][i] - (B0 + B1x*x + B1y*y + B2x*x*x + B2y*y*y+ B3x*x*x*x + B3y*y*y*y))**2
        err = err/lenX/lenY
        return err

    lenX = len(B[0])
    lenY = len(B)
    B = np.array(B)
    x0 = [30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    res = minimize(_full_error, x0, ((X,Y), B), 'Powell', tol=1e-6)
    print(res)
    B0, B1x, B1y, B2x, B2y, B3x, B3y = res.x
    return B - np.array([[B0 + B1x*X[j][i] + B1y*Y[j][i]+ B2x*X[j][i]*X[j][i] + B2y*Y[j][i]*Y[j][i] + B3x*(X[j][i]**3) + B3y*(Y[j][i]**3) for i in range(lenX)] for j in range(lenY)])
#
def slowCutter(data):
    """ cut slow field, approximated by polynomes""" 
    return v3f.vec3Field(data.X,data.Y,
                     cutSlowProcesser(data.Bx,data.X,data.Y),
                     cutSlowProcesser(data.By,data.X,data.Y),
                     cutSlowProcesser(data.Bz,data.X,data.Y),
                     data.vol,data.steps)
#

def autoRecRegionCutter(field):
    """automaticaly find recatangle region, by maximal values of Bz and return this region"""
    res = v3f.magnToPicture(np.abs(field.Bz))
    maxMassive = peak_local_max(np.abs(res),min_distance=1)
    #print(maxMassive)
    if(len(maxMassive)>4):
        maxX = max(list(zip(*maxMassive))[1])
        minX = min(list(zip(*maxMassive))[1])
        maxY = max(list(zip(*maxMassive))[0])
        minY = min(list(zip(*maxMassive))[0])
        tempX =[elem[minX:maxX] for elem in field.Bx[minY:maxY]] 
        tempY =[elem[minX:maxX] for elem in field.By[minY:maxY]]
        tempZ =[elem[minX:maxX] for elem in field.Bz[minY:maxY]]
        
        X1, Y1  = np.meshgrid(np.arange(minX*field.steps[1]+field.vol[1],maxX*field.steps[1]+field.vol[1],field.steps[1]), 
                              np.arange(minY*field.steps[0]+field.vol[0],maxY*field.steps[0]+field.vol[0],field.steps[0])
                              )
        return v3f.vec3Field(X1,Y1,tempX,tempY,tempZ,
                             [minX*field.steps[1]+field.vol[1],minY*field.steps[0]+field.vol[0],field.vol[2],
                              (maxX-minX-1)*(field.steps[1]),(maxY-minY-1)*(field.steps[0]),0],field.steps
                              )
    else:
        print("too small points to analyse")
        return field
