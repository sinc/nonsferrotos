import  numpy as np
import  scipy.optimize as opt

from . import vec3Field as v3f
from ..models.recAnalitic import B_calc_a,P_calc_a,T_calc_a,R_calc_a,EulerZ

#
def modelZField(X,Y, vol,steps,sample, M0, a, B0, theta):
    return v3f.vec3Field(X,Y,[],[],
                         [[B_calc_a(X[j][i],Y[j][i],vol[2],sample,M0,a,B0,theta) for i in range(len(X[0]))] for j in range(len(X))],
                         vol,steps)
#
def solveMagnetization(field,sample,thetta):
    xSize = len(field.X[0])
    ySize = len(field.X)
    A = np.zeros((xSize*ySize, 12));
    b = [0.0]*(xSize*ySize);
    z = field.vol[2];
    for i in range(xSize):
        for j in range(ySize):
            b[i + j * xSize] = field.Bz[j][i]
            x = field.X[j][i]
            y = field.Y[j][i]
            tempP = P_calc_a(x, y, z,sample)
            tempR = R_calc_a(x, y, z,sample)
            tempT = T_calc_a(x, y, z,sample)
            for l in range(3):
                A[int(i + j * xSize)][l] = -EulerZ(tempR[0][l], tempR[1][l], tempR[2][l],thetta) #M[0],M[1],M[2] signs!!
            #signs!!!
            A[int(i + j * xSize)][ 3] = (EulerZ(tempP[0], tempP[1], tempP[2],thetta) - EulerZ(tempT[0][0][0], tempT[1][0][0], tempT[2][0][0],thetta))       # a[0] 
            A[int(i + j * xSize)][ 4] = -(EulerZ(tempT[0][0][1], tempT[1][0][1], tempT[2][0][1],thetta) + EulerZ(tempT[0][1][0], tempT[1][1][0], tempT[2][1][0],thetta)) # a[1]
            A[int(i + j * xSize)][ 5] = -(EulerZ(tempT[0][0][2], tempT[1][0][2], tempT[2][0][2],thetta) + EulerZ(tempT[0][2][0], tempT[1][2][0], tempT[2][2][0],thetta)) # a[2]
            A[int(i + j * xSize)][ 6] = (EulerZ(tempP[0], tempP[1], tempP[2],thetta) - EulerZ(tempT[0][1][1], tempT[1][1][1], tempT[2][1][1],thetta))       # a[3]
            A[int(i + j * xSize)][ 7] = -(EulerZ(tempT[0][1][2], tempT[1][1][2], tempT[2][1][2],thetta) + EulerZ(tempT[0][2][1], tempT[1][2][1], tempT[2][2][1],thetta)) # a[4]
            A[int(i + j * xSize)][ 8] = (EulerZ(tempP[0], tempP[1], tempP[2],thetta) - EulerZ(tempT[0][2][2], tempT[1][2][2], tempT[2][2][2],thetta))       # a[5]
            A[int(i + j * xSize)][ 9 + 0] = 1;#B0[0]
            A[int(i + j * xSize)][ 9 + 1] = x;#B0[1]
            A[int(i + j * xSize)][ 9 + 2] = y;#B0[2]
    return np.linalg.lstsq(A, b, 1e-20) #[M0[0],M0[1],M0[2],M1[0],M1[1],M1[2],M1[3],M1[4],M1[5],B0[0],B0[1],B0[2]]  
#
def errFunc(sample,field):
    thetta = [0.0]*3
    magnModel = solveMagnetization(field,sample,thetta)
    print(magnModel)
    retVal = sum(np.array(field.Bz)-np.array(modelZField(field.X,field.Y,field.vol,field.steps,
                                                       sample,magnModel[:3],magnModel[3:9],magnModel[9:],thetta)
                                           )
               )
    return retVal
#
def solveFormAndMagnetization(field,initialSample,regionSize):
    print(errFunc(initialSample,field),initialSample)
    result = opt.minimize(lambda xv: errFunc([xv[0],xv[1],xv[2],xv[3],xv[4],initialSample[5]],field),initialSample[:5],method = 'SLSQP',bounds = regionSize)
    print(result)
    newSample = [result['x'][0],result['x'][1],result['x'][2],result['x'][3],result['x'][4],initialSample[5]]
    return newSample, solveMagnetization(field, newSample, [0.0]*3) 
def magnetCutter(field,initialSample,regionSize):
    form, magn =  solveFormAndMagnetization(field,initialSample,regionSize)
    retField = v3f.vec3Field(field.X,field.Y,[],[],
                             np.array(field.Bz)-np.array(
                                 modelZField(field.X,field.Y,field.vol,field.steps,form, magnModel[:3],magnModel[3:9],magnModel[9:],[0,0]*3).Bz),
                             field.vol,field.steps)