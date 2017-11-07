import numpy as np
def Atanh(x):
    if(np.abs(x) > 1):
        print("x")
        return 0
    return 0.5 * np.log((1 + x) / (1 - x))
def EulerZ(Bx,  By,  Bz, Theta):
    return -np.sin(Theta[1]) * Bx + np.cos(Theta[1]) * np.sin(Theta[0]) * By + np.cos(Theta[0]) * np.cos(Theta[1]) * Bz
def EulerX(Bx,  By,  Bz, Theta):
    return np.cos(Theta[1]) * np.cos(Theta[2]) * Bx +(np.cos(Theta[2]) * np.sin(Theta[0]) * np.sin(Theta[1]) - np.cos(Theta[0]) * np.sin(Theta[2])) * By +(np.cos(Theta[0]) * np.cos(Theta[2]) * np.sin(Theta[1]) + np.sin(Theta[0]) * np.sin(Theta[2])) * Bz
def EulerY(Bx,  By,  Bz, Theta):
    return np.cos(Theta[1]) * np.sin(Theta[2]) * Bx +(np.sin(Theta[0]) * np.sin(Theta[1]) * np.sin(Theta[2]) + np.cos(Theta[0]) * np.cos(Theta[2])) * By +(np.cos(Theta[0]) * np.sin(Theta[1]) * np.sin(Theta[2]) - np.sin(Theta[0]) * np.cos(Theta[2])) * Bz
def Pz( x,  y,  z,  x1,  y1,  z1):
    if (np.abs(z - z1) < 1e-15):
        return y1 + (y - y1) * np.log(x - x1 + np.sqrt((x - x1) * (x - x1) + (z - z1) * (z - z1) + (y - y1) * (y - y1))) +(x - x1) * np.log(y - y1 + np.sqrt((x - x1) * (x - x1) + (z - z1) * (z - z1) + (y - y1) * (y - y1)))
    else:
        return y1 + (z - z1) * np.arctan((y - y1) / (z - z1)) -(z - z1) * np.arctan(((x - x1) * (y - y1)) / ((z - z1) * np.sqrt((x - x1) * (x - x1) + (z - z1) * (z - z1) + (y - y1) * (y - y1)))) +(y - y1) * np.log(x - x1 + np.sqrt((x - x1) * (x - x1) + (z - z1) * (z - z1) + (y - y1) * (y - y1))) +(x - x1) * np.log(y - y1 + np.sqrt((x - x1) * (x - x1) + (z - z1) * (z - z1) + (y - y1) * (y - y1)))
def Px( x,  y,  z,  x1,  y1,  z1):
    value = Pz(z, y, x, z1, y1, x1)
    return value
def Py( x,  y,  z,  x1,  y1,  z1):
    return Pz(x, z, y, x1, z1, y1)
def Qzx( x,  y,  z,  x1,  y1,  z1):
    return -(x - x1) * x1 * np.log((y - y1) + np.sqrt((y - y1) * (y - y1) + (z - z1) + (x - x1) * (x - x1))) + (-x) * (-x1 - (z - z1) * np.arctan((x - x1) / (z - z1)) +
        (z - z1) * np.arctan(((y - y1) * (x - x1)) / ((z - z1) * np.sqrt((y - y1) * (y - y1) + (z - z1) + (x - x1) * (x - x1)))) -
        (x - x1) * np.log((y - y1) + np.sqrt((y - y1) * (y - y1) + (z - z1) + (x - x1) * (x - x1))) -
        (y - y1) * np.log(x + np.sqrt((y - y1) * (y - y1) + (z - z1) + (x - x1) * (x - x1)) - x1))
def Qzy( x,  y,  z,  x1,  y1,  z1):
    return Qzx(y, x, z, y1, x1, z1)
def Qzz( x,  y,  z,  x1,  y1,  z1):
    return (-(1.0 / 2.0)) * (z - z1) * (z - z1) * np.arctan(((x - x1) * (y - y1)) / (np.sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1) + (z - z1) * (z - z1)) * (z - z1))) -((x - x1) * (y - y1) * np.sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1) + (z - z1) * (z - z1)) * (z - z1) -
        (((x - x1) * (x - x1) * ((y - y1) * (y - y1) + (z - z1) * (z - z1)) + (y - y1) * (y - y1) * (z - z1) * (z - z1)) *
        np.arctan((np.sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1) + (z - z1) * (z - z1)) * (z - z1)) / ((x - x1) * (y - y1))))) / (2.0 * (z - z1) * (z - z1)) +z * ((z - z1) * np.arctan(((x - x1) * (y - y1)) / (np.sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1) + (z - z1) * (z - z1)) * (z - z1))) +
        (1.0 / (z - z1)) * (-((np.sqrt((x - x1) * (x - x1) * ((y - y1) * (y - y1) + (z - z1) * (z - z1)) + (y - y1) * (y - y1) * (z - z1) * (z - z1)) *
        Atanh(((x - x1) * (y - y1) * (z - z1)) / (np.sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1) + (z - z1) * (z - z1)) *
        np.sqrt((x - x1) * (x - x1) * ((y - y1) * (y - y1) + (z - z1) * (z - z1)) + (y - y1) * (y - y1) *
        (z - z1) * (z - z1)))))) +
        (x - x1) * (y - y1) * np.log((z - z1) + np.sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1) + (z - z1) * (z - z1)))))
def Rzx( x,  y,  z,  x1,  y1,  z1):
    return -np.log(y - y1 + np.sqrt(x * x - 2.0* x * x1 + x1 * x1 + (y - y1) * (y - y1) + z * z - 2 * z * z1 + z1 * z1))
def Rzy( x,  y,  z,  x1,  y1,  z1):
    return Rzx(y, x, z, y1, x1, z1)
def Rzz( x,  y,  z,  x1,  y1,  z1):
    if (np.abs(z-z1) < 1e-15 and np.abs((x - x1) * (y - y1)) < 1e-15):
        return 0
    return -np.arctan(((x - x1) * (y - y1)) / ((z - z1) *
        np.sqrt((x - x1) * (x - x1) + (z - z1) * (z - z1) + (y - y1) * (y - y1))))
def Rxz( x,  y,  z,  x1,  y1,  z1):
    return Rzx(z, y, x, z1, y1, x1)
def Rxy( x,  y,  z,  x1,  y1,  z1):
    return Rzy(z, y, x, z1, y1, x1)
def Rxx( x,  y,  z,  x1,  y1,  z1):
    return Rzz(z, y, x, z1, y1, x1)
def Ryz( x,  y,  z,  x1,  y1,  z1):
    return Rzy(x, z, y, x1, z1, y1)
def Ryx( x,  y,  z,  x1,  y1,  z1):
    return Rxy(y, x, z, y1, x1, z1)
def Ryy( x,  y,  z,  x1,  y1,  z1):
    return Rzz(x, z, y, x1, z1, y1)
def Tzxx( x,  y,  z,  x1,  y1,  z1):
    return -x1 * np.log(y - y1 + np.sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1) + (z - z1) * (z - z1)))
def Tzxy( x,  y,  z,  x1,  y1,  z1):
    return ((x - x1) * (x - x1) + (y - y1) * (y - y1) + (z - z1) * (z - z1) -
        y * np.sqrt(x * x - 2.0 * x * x1 + x1 * x1 + y * y - 2 * y * y1 + y1 * y1 + z * z - 2.0 * z * z1 + z1 * z1) *
        np.log(y - y1 + np.sqrt(x * x - 2 * x * x1 + x1 * x1 + (y - y1) * (y - y1) + z * z - 2.0 * z * z1 + z1 * z1))) /np.sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1) + (z - z1) * (z - z1))
def Tzxz( x,  y,  z,  x1,  y1,  z1):
    if (np.abs(z - z1) < 1e-15 and np.abs((x - x1) * (y - y1)) < 1e-15):
        return 0
    return np.arctan(((y - y1) * (x - x1)) / ((z - z1) * np.sqrt((y - y1) * (y - y1) + (z - z1) * (z - z1) + (x - x1) * (x - x1)))) * z1
def Tzyx( x,  y,  z,  x1,  y1,  z1):
    return Tzxy(y, x, z, y1, x1, z1)
def Tzyy( x,  y,  z,  x1,  y1,  z1):
    return Tzxx(y, x, z, y1, x1, z1)
def Tzyz( x,  y,  z,  x1,  y1,  z1):
    return Tzxz(y, x, z, y1, x1, z1)
def Tzzx( x,  y,  z,  x1,  y1,  z1):
    if (np.abs(z - z1) < 1e-15 and np.abs((x - x1) * (y - y1)) < 1e-15):
        return 0
    return ((x * np.arctan(((y - y1) * (x - x1)) / ((z - z1) *
        np.sqrt((y - y1) * (y - y1) + (z - z1) * (z - z1) + (x - x1) * (x - x1)))))  +
            (z - z1) * np.log((y - y1) + np.sqrt((y - y1) * (y - y1) + (z - z1) * (z - z1) + (x - x1) * (x - x1))))
def Tzzy( x,  y,  z,  x1,  y1,  z1):
    return Tzzx(y, x, z, y1, x1, z1)
def Tzzz( x,  y,  z,  x1,  y1,  z1):
    if (np.abs(z - z1) < 1e-15 and np.abs((x - x1) * (y - y1)) < 1e-15):
        return 0
    return np.arctan(((x - x1) * (y1 - y)) /
        ((z - z1) * np.sqrt((z - z1) * (z - z1) + (y - y1) * (y - y1) + (x - x1) * (x - x1)))) * z1
def Txxx( x,  y,  z,  x1,  y1,  z1):
    return Tzzz(z, y, x, z1, y1, x1)
def Txxy( x,  y,  z,  x1,  y1,  z1):
    return Tzzy(z, y, x, z1, y1, x1)
def Txxz( x,  y,  z,  x1,  y1,  z1):
    return Tzzx(z, y, x, z1, y1, x1)
def Txyx( x,  y,  z,  x1,  y1,  z1):
    return Tzyz(z, y, x, z1, y1, x1)
def Txyy( x,  y,  z,  x1,  y1,  z1):
    return Tzyy(z, y, x, z1, y1, x1)
def Txyz( x,  y,  z,  x1,  y1,  z1):
    return Tzyx(z, y, x, z1, y1, x1)
def Txzx( x,  y,  z,  x1,  y1,  z1):
    return Tzxz(z, y, x, z1, y1, x1)
def Txzy( x,  y,  z,  x1,  y1,  z1):
    return Tzxy(z, y, x, z1, y1, x1)
def Txzz( x,  y,  z,  x1,  y1,  z1):
    return Tzxx(z, y, x, z1, y1, x1)
def Tyxx( x,  y,  z,  x1,  y1,  z1):
    return Tzxx(x, z, y, x1, z1, y1)
def Tyxy( x,  y,  z,  x1,  y1,  z1):
    return Tzxz(x, z, y, x1, z1, y1)
def Tyxz( x,  y,  z,  x1,  y1,  z1):
    return Tzxy(x, z, y, x1, z1, y1)
def Tyyx( x,  y,  z,  x1,  y1,  z1):
    return Tzzx(x, z, y, x1, z1, y1)
def Tyyy( x,  y,  z,  x1,  y1,  z1):
    return Tzzz(x, z, y, x1, z1, y1)
def Tyyz( x,  y,  z,  x1,  y1,  z1):
    return Tzzy(x, z, y, x1, z1, y1)
def Tyzx( x,  y,  z,  x1,  y1,  z1):
    return Tzyx(x, z, y, x1, z1, y1)
def Tyzy( x,  y,  z,  x1,  y1,  z1):
    return Tzyz(x, z, y, x1, z1, y1)
def Tyzz( x,  y,  z,  x1,  y1,  z1):
    return Tzyy(x, z, y, x1, z1, y1)

def limits_3D(x,  y,  z, sample, func):
    return func(x, y, z, sample[0] + sample[3], sample[1] + sample[4], 0)- func(x, y, z, sample[0] + sample[3], sample[1], 0)  + func(x, y, z, sample[0] + sample[3], sample[1], -sample[5])- func(x, y, z, sample[0] + sample[3], sample[1] + sample[4], -sample[5]) + func(x, y, z, sample[0], sample[1] + sample[4], -sample[5]) - func(x, y, z, sample[0], sample[1] + sample[4], 0) + func(x, y, z, sample[0], sample[1], 0) - func(x, y, z, sample[0], sample[1], -sample[5])
def limits_2D_normX( x,  y,  z,  Xcoord, sample, func):
    value = func(x, y, z, Xcoord, sample[1] + sample[4], 0) - func(x, y, z, Xcoord, sample[1] + sample[4], -sample[5]) + func(x, y, z, Xcoord, sample[1], -sample[5]) - func(x, y, z, Xcoord, sample[1], 0)
    return value
def limits_2D_normY( x,  y,  z,  Ycoord, sample, func):
	 return func(x, y, z, sample[0] + sample[3], Ycoord, 0) - func(x, y, z, sample[0] + sample[3], Ycoord, -sample[5]) + func(x, y, z, sample[0], Ycoord, -sample[5]) - func(x, y, z, sample[0], Ycoord, 0)
def limits_2D_normZ( x,  y,  z,  Zcoord, sample, func):
    return func(x, y, z, sample[0] + sample[3], sample[1] + sample[4], Zcoord) - func(x, y, z, sample[0] + sample[3], sample[1], Zcoord)  + func(x, y, z, sample[0], sample[1], Zcoord) - func(x, y, z, sample[0], sample[1] + sample[4], Zcoord)
def limits_2D_surfX( x,  y,  z, sample, func):
    return limits_2D_normX(x, y, z, sample[0] + sample[3], sample, func) - limits_2D_normX(x, y, z, sample[0], sample, func)
def limits_2D_surfY( x,  y,  z, sample, func):
    return limits_2D_normY(x, y, z, sample[1] + sample[4], sample, func) - limits_2D_normY(x, y, z, sample[1], sample, func)
def limits_2D_surfZ( x,  y,  z, sample, func):
    return -limits_2D_normZ(x, y, z, -sample[5], sample, func) + limits_2D_normZ(x, y, z, 0, sample, func)

def P_calc_a(x, y, z, sample):
    return [limits_3D(x, y, z, sample, Px), limits_3D(x, y, z, sample, Py), limits_3D(x, y, z, sample, Pz)] 
def R_calc_a(x, y, z, sample):
    return [
     [limits_2D_surfX(x, y, z, sample, Rxx), limits_2D_surfY(x, y, z, sample, Rxy), limits_2D_surfZ(x, y, z, sample, Rxz)] ,
     [limits_2D_surfX(x, y, z, sample, Ryx), limits_2D_surfY(x, y, z, sample, Ryy), limits_2D_surfZ(x, y, z, sample, Ryz)] ,
     [limits_2D_surfX(x, y, z, sample, Rzx), limits_2D_surfY(x, y, z, sample, Rzy), limits_2D_surfZ(x, y, z, sample, Rzz)]  
     ]
def T_calc_a(x, y, z, sample):
    return [
    	  [   [limits_2D_surfX(x, y, z, sample, Txxx), limits_2D_surfX(x, y, z, sample, Txxy), limits_2D_surfX(x, y, z, sample, Txxz)] ,
		     [limits_2D_surfY(x, y, z, sample, Txyx), limits_2D_surfY(x, y, z, sample, Txyy), limits_2D_surfY(x, y, z, sample, Txyz)] ,
		     [limits_2D_surfZ(x, y, z, sample, Txzx), limits_2D_surfZ(x, y, z, sample, Txzy), limits_2D_surfZ(x, y, z, sample, Txzz)] ,
	    ],
        [
		     [limits_2D_surfX(x, y, z, sample, Tyxx), limits_2D_surfX(x, y, z, sample, Tyxy), limits_2D_surfX(x, y, z, sample, Tyxz)] ,
		     [limits_2D_surfY(x, y, z, sample, Tyyx), limits_2D_surfY(x, y, z, sample, Tyyy), limits_2D_surfY(x, y, z, sample, Tyyz)] ,
		     [limits_2D_surfZ(x, y, z, sample, Tyzx), limits_2D_surfZ(x, y, z, sample, Tyzy), limits_2D_surfZ(x, y, z, sample, Tyzz)] ,
	    ],
        [
		     [limits_2D_surfX(x, y, z, sample, Tzxx), limits_2D_surfX(x, y, z, sample, Tzxy), limits_2D_surfX(x, y, z, sample, Tzxz)] ,
		     [limits_2D_surfY(x, y, z, sample, Tzyx), limits_2D_surfY(x, y, z, sample, Tzyy), limits_2D_surfY(x, y, z, sample, Tzyz)] ,
		     [limits_2D_surfZ(x, y, z, sample, Tzzx), limits_2D_surfZ(x, y, z, sample, Tzzy), limits_2D_surfZ(x, y, z, sample, Tzzz)] ,
	    ]
        ]

def B_calc_a(x,  y,  z, sample, M0, a, B0, theta):
    """it can be commmented and a[] and b[] can be used directly to increasing speed"""
    aMatrix = [	
		 [a[0], a[1], a[2]],
		 [a[1], a[3], a[4]],
		 [a[2], a[4], a[5]]
         ] 
    tempP = P_calc_a(x, y, z, sample)
    tempR = R_calc_a(x, y, z, sample)
    tempT = T_calc_a(x, y, z, sample)
    Bx = (aMatrix[0, 0] + aMatrix[1, 1] + aMatrix[2, 2]) * tempP[0]
    By = (aMatrix[0, 0] + aMatrix[1, 1] + aMatrix[2, 2]) * tempP[1]
    Bz = (aMatrix[0, 0] + aMatrix[1, 1] + aMatrix[2, 2]) * tempP[2]
    for i in range(3):
        Bx += -M0[i] * tempR[0, i]
        By += -M0[i] * tempR[1, i]
        Bz += -M0[i] * tempR[2, i]
        for l in range(3):
            Bx += -aMatrix[i, l] * tempT[0, i, l]
            #//for (int k = 0 k < 3 k++) ret += -bMatrix[i, l, k] * tempY[i, l, k]
            By += -aMatrix[i, l] * tempT[1, i, l]
            #//for (int k = 0 k < 3 k++) ret += -bMatrix[i, l, k] * tempY[i, l, k]
            Bz += -aMatrix[i, l] * tempT[2, i, l]
            #//for (int k = 0 k < 3 k++) ret += -bMatrix[i, l, k] * tempY[i, l, k]
    return -Math.Sin(theta[1]) * Bx + Math.Cos(theta[1]) * Math.Sin(theta[0]) * By + Math.Cos(theta[0]) * Math.Cos(theta[1]) * Bz + B0[0] + B0[1] * x + B0[2] * y
#
def modelZField(X,Y, vol,steps,sample, M0, a, B0, theta):
    return v3f.vec3Field(X,Y,[],[],
                         [[B_calc_a(X[j][i],Y[j][i],vol[2],sample,M0,a,B0,theta) for i in range(len(X[0]))] for j in range(len(X))],
                         vol,steps)
 
def solveMagnetization(field,sample,thetta):
    A = np.zeros((m_field.dimX * m_field.dimY, 12));
    b = [m_field.dimX * m_field.dimY]*0.0;
    z = field.vol[2];
    for i in range(len(field.X[0])):
        for j in range(len(field.X)):
            b[i + j * m_field.dimX] = m_field.getField(i + j * m_field.dimX, 2)
            x = X[j][i]
            y = Y[j][i]
            tempP = P_calc_a(x, y, z)
            tempR = R_calc_a(x, y, z)
            tempT = T_calc_a(x, y, z)
            for l in range(3):
                A[i + j * m_field.dimX, l] = -EulerZ(tempR[0, l], tempR[1, l], tempR[2, l],thetta) #M[0],M[1],M[2] signs!!
            #signs!!!
            A[i + j * m_field.dimX, 3] = (EulerZ(tempP[0], tempP[1], tempP[2],thetta) - EulerZ(tempT[0, 0, 0], tempT[1, 0, 0], tempT[2, 0, 0],thetta))       # a[0] 
            A[i + j * m_field.dimX, 4] = -(EulerZ(tempT[0, 0, 1], tempT[1, 0, 1], tempT[2, 0, 1],thetta) + EulerZ(tempT[0, 1, 0], tempT[1, 1, 0], tempT[2, 1, 0],thetta)) # a[1]
            A[i + j * m_field.dimX, 5] = -(EulerZ(tempT[0, 0, 2], tempT[1, 0, 2], tempT[2, 0, 2],thetta) + EulerZ(tempT[0, 2, 0], tempT[1, 2, 0], tempT[2, 2, 0],thetta)) # a[2]
            A[i + j * m_field.dimX, 6] = (EulerZ(tempP[0], tempP[1], tempP[2],thetta) - EulerZ(tempT[0, 1, 1], tempT[1, 1, 1], tempT[2, 1, 1],thetta))       # a[3]
            A[i + j * m_field.dimX, 7] = -(EulerZ(tempT[0, 1, 2], tempT[1, 1, 2], tempT[2, 1, 2],thetta) + EulerZ(tempT[0, 2, 1], tempT[1, 2, 1], tempT[2, 2, 1],thetta)) # a[4]
            A[i + j * m_field.dimX, 8] = (EulerZ(tempP[0], tempP[1], tempP[2],thetta) - EulerZ(tempT[0, 2, 2], tempT[1, 2, 2], tempT[2, 2, 2],thetta))       # a[5]
            A[i + j * m_field.dimX, 9 + 0] = 1;#B0[0]
            A[i + j * m_field.dimX, 9 + 1] = x;#B0[1]
            A[i + j * m_field.dimX, 9 + 2] = y;#B0[2]
    return np.linalg.lstsq(A, b, 1e-20) #[M0[0],M0[1],M0[2],M1[0],M1[1],M1[2],M1[3],M1[4],M1[5],B0[0],B0[1],B0[2]]  
def errFunc(sample,field):
    thetta = [0.0]*3
    magnModel = solveMagnetization(field,sample,thetta)
    print('allSolution:',magnModel)
    magnModel = magnModel['X']
    print('X:',magnModel)
    retVal = sum(np.array(field.Bz)-np.array(modelZField(field.X,field.Y,field.vol,field.steps,
                                                       sample,magnModel[:3],magnModel[3:9],magnModel[9:],thetta)
                                           )
               )
    return retVal
def solveFormAndMagnetization(field,initialSample,regionSize):
    result = opt.minimize(lambda xv: errFunc([xv[0],xv[1],xv[2],xv[3],xv[4],initialSample[5]],field),initialSample[:5],method = 'SLSQP',bounds = regionSize)
    print(result)
    return solveMagnetization(field, result['x'], [0.0]*3)