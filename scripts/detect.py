#!/usr/bin/env pytnon
#
import sys
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from scipy import arange
from mpl_toolkits.mplot3d import axes3d
from numpy.fft import *

from scipy.ndimage.interpolation import *

def rms(Bz, sx, sy):
    lenX = len(Bz[0])
    lenY = len(Bz)
    res = []
    for l in range(lenY-sy):
        t1 = []
        for k in range(lenX-sx):
            res_den = 0.0
            for j in range(sy):
                for i in range(sx):
                    res_den += (Bz[l][k] - Bz[l+j][k+i])**2
            t1.append(res_den)
        res.append(t1)
    return res

def spectrumDetector(B, sx, sy, K):
    lenX = len(B[0])
    lenY = len(B)
    return [[sum(np.abs(fft2(B[j:j+sy,i:i+sx]))[1:K, 1:K].ravel()) for i in range(int(lenX-sx))] for j in range(int(lenY-sy))]

def correlate2D(B1, B2, lenX, lenY):
    aux = np.zeros((3*lenX-2,3*lenY-2))
    for j in range(lenY):
        for i in range(lenX):
            aux[j+lenY-1][i+lenX-1] = B1[j][i]
    result = 0.0
    for jj in range(2*lenY-1):
        for ii in range(2*lenX-1):
            for j in range(lenY):
                for i in range(lenX):
                    result += aux[j+jj][i+ii] * B2[j][i]
    return result

def corrDectector1(B, sx, sy):
    lenX = len(B[0])
    lenY = len(B)
    aux = np.zeros((3*sx-2,3*sy-2))
    result1 = []
    result2 = []
    for l in range(lenY-sy):
        t1 = []
        t2 = []
        for k in range(lenX-sx):
            res_num1 = 0.0
            res_num2 = 0.0
            res_den = 0.0
            for j in range(sy):
                for i in range(sx):
                    aux[j+sy-1][i+sx-1] = B[l+j][k+i]
            for jj in range(2*sy-1):
                for ii in range(2*sx-1):
                    for j in range(sy):
                        for i in range(sx):
                            res_den += aux[j+jj][i+ii] * B[j][i]
                            res_num1 += (float(ii)-float(sx)/2.0)*aux[j+jj][i+ii] * B[l+j][k+i]
                            res_num2 += (float(jj)-float(sy)/2.0)*aux[j+jj][i+ii] * B[l+j][k+i]
            t1.append(res_num1/res_den)
            t2.append(res_num2/res_den)
        result1.append(t1)
        result2.append(t2)
    return result1, result2

def corrDectector(Bx, By, Bz, sx, sy):
    lenX = len(Bx[0])
    lenY = len(Bx)
    auxX = np.zeros((3*sx-2,3*sy-2))
    auxY = np.zeros((3*sx-2,3*sy-2))
    auxZ = np.zeros((3*sx-2,3*sy-2))
    result1 = []
    result2 = []
    for l in range(lenY-sy):
        t1 = []
        t2 = []
        for k in range(lenX-sx):
            res_num1 = 0.0
            res_num2 = 0.0
            res_den = 0.0
            for j in range(sy):
                for i in range(sx):
                    auxX[j+sy-1][i+sx-1] = Bx[l+j][k+i]
                    auxY[j+sy-1][i+sx-1] = By[l+j][k+i]
                    auxZ[j+sy-1][i+sx-1] = Bz[l+j][k+i]
            for jj in range(2*sy-1):
                for ii in range(2*sx-1):
                    for j in range(sy):
                        for i in range(sx):
                            res_den += auxX[j+jj][i+ii] * Bx[l+j][k+i] + auxY[j+jj][i+ii] * By[l+j][k+i] + auxZ[j+jj][i+ii] * Bz[l+j][k+i]
                            res_num1 += ((float(ii)-float(sx)/2.0)**2.0+(float(jj)-float(sy)/2.0)**2.0)*(auxX[j+jj][i+ii] * Bx[l+j][k+i] + auxY[j+jj][i+ii] * By[l+j][k+i] + auxZ[j+jj][i+ii] * Bz[l+j][k+i])
                            res_num2 += (float(jj)-float(sy)/2.0)*(auxX[j+jj][i+ii] * Bx[l+j][k+i] + auxY[j+jj][i+ii] * By[l+j][k+i] + auxZ[j+jj][i+ii] * Bz[l+j][k+i])
            t1.append(res_num1/res_den)
            t2.append(res_num2/res_den)
        result1.append(t1)
        result2.append(t2)
    return result1, result2

def corrDectector2(Bx, By, Bz, sx, sy):
    lenX = len(Bx[0])
    lenY = len(Bx)
    result1 = []
    result2 = []
    for l in range(lenY-sy):
        t1 = []
        t2 = []
        for k in range(lenX-sx):
            res_num1 = 0.0
            res_num2 = 0.0
            res_den = 0.0
            for j in range(sy):
                for i in range(sx):
                    res_den += Bx[l][k] * Bx[l+j][k+i] + By[l][k] * By[l+j][k+i] + Bz[l][k] * Bz[l+j][k+i]
                    res_num1 += ((float(i)-float(sx)/2.0)**2.0+(float(j)-float(sy)/2.0)**2.0)*(Bx[l][k] * Bx[l+j][k+i] + By[l][k] * By[l+j][k+i] + Bz[l][k] * Bz[l+j][k+i])
                    #res_num1 += (float(i)-float(sx)/2.0)*(Bx[l][k] * Bx[l+j][k+i] + By[l][k] * By[l+j][k+i] + Bz[l][k] * Bz[l+j][k+i])
                    #res_num2 += (float(j)-float(sy)/2.0)*(Bx[l][k] * Bx[l+j][k+i] + By[l][k] * By[l+j][k+i] + Bz[l][k] * Bz[l+j][k+i])
            t1.append(res_num1/res_den)
            t2.append(res_num2/res_den)
        result1.append(t1)
        result2.append(t2)
    return result1, result2

def corrDectector3(Bz, sx, sy):
    lenX = len(Bz[0])
    lenY = len(Bz)
    result1 = []
    result2 = []
    for l in range(lenY-sy):
        t1 = []
        t2 = []
        for k in range(lenX-sx):
            res_num1 = 0.0
            res_num2 = 0.0
            res_den = 0.0
            for j in range(sy):
                for i in range(sx):
                    res_den += Bz[l][k] * Bz[l+j][k+i]
                    res_num1 += (float(i)-float(sx)/2.0)*(Bz[l][k] * Bz[l+j][k+i])
                    #res_num1 += ((float(i)-float(sx)/2.0)**2.0+(float(j)-float(sy)/2.0)**2.0)*(Bz[l][k] * Bz[l+j][k+i])
                    #res_num1 += (float(i)-float(sx)/2.0)*(Bx[l][k] * Bx[l+j][k+i] + By[l][k] * By[l+j][k+i] + Bz[l][k] * Bz[l+j][k+i])
                    #res_num2 += (float(j)-float(sy)/2.0)*(Bx[l][k] * Bx[l+j][k+i] + By[l][k] * By[l+j][k+i] + Bz[l][k] * Bz[l+j][k+i])
            t1.append(res_num1/res_den)
            t2.append(res_num2/res_den)
        result1.append(t1)
        result2.append(t2)
    return result1, result2

def laplasDetector(B, sx = 2, sy = 2):
    lenX = len(B[0])
    lenY = len(B)
    # 1  1  1
    # 1 -8  1
    # 1  1  1
    result = []
    for l in range(lenY-2*sy):
        t = []
        for k in range(lenX-2*sx):
            t.append(B[l][k] + B[l][k+sx] + B[l][k+2*sx] + B[l+sy][k] + B[l+sy][k+2*sx] + B[l+2*sy][k] + B[l+2*sy][k+sx] + B[l+2*sy][k+2*sx] - 8*B[l+sy][k+sx])
        result.append(t)
    return result


def spectrumDetector(B, sx, sy, K):
    lenX = len(B[0])
    lenY = len(B)
    return [[sum(np.abs(fft2(B[j:j+sy,i:i+sx]))[1:K, 1:K].ravel()) for i in range(int(lenX-sx))] for j in range(int(lenY-sy))]

def correlate2D(B1, B2, lenX, lenY):
    aux = np.zeros((3*lenX-2,3*lenY-2))
    for j in range(lenY):
        for i in range(lenX):
            aux[j+lenY-1][i+lenX-1] = B1[j][i]
    result = 0.0
    for jj in range(2*lenY-1):
        for ii in range(2*lenX-1):
            for j in range(lenY):
                for i in range(lenX):
                    result += aux[j+jj][i+ii] * B2[j][i]
    return result

def corrDectector1(B, sx, sy):
    lenX = len(B[0])
    lenY = len(B)
    aux = np.zeros((3*sx-2,3*sy-2))
    result1 = []
    result2 = []
    for l in range(lenY-sy):
        t1 = []
        t2 = []
        for k in range(lenX-sx):
            res_num1 = 0.0
            res_num2 = 0.0
            res_den = 0.0
            for j in range(sy):
                for i in range(sx):
                    aux[j+sy-1][i+sx-1] = B[l+j][k+i]
            for jj in range(2*sy-1):
                for ii in range(2*sx-1):
                    for j in range(sy):
                        for i in range(sx):
                            res_den += aux[j+jj][i+ii] * B[j][i]
                            res_num1 += (float(ii)-float(sx)/2.0)*aux[j+jj][i+ii] * B[l+j][k+i]
                            res_num2 += (float(jj)-float(sy)/2.0)*aux[j+jj][i+ii] * B[l+j][k+i]
            t1.append(res_num1/res_den)
            t2.append(res_num2/res_den)
        result1.append(t1)
        result2.append(t2)
    return result1, result2

def corrDectector(Bx, By, Bz, sx, sy):
    lenX = len(Bx[0])
    lenY = len(Bx)
    auxX = np.zeros((3*sx-2,3*sy-2))
    auxY = np.zeros((3*sx-2,3*sy-2))
    auxZ = np.zeros((3*sx-2,3*sy-2))
    result1 = []
    result2 = []
    for l in range(lenY-sy):
        t1 = []
        t2 = []
        for k in range(lenX-sx):
            res_num1 = 0.0
            res_num2 = 0.0
            res_den = 0.0
            for j in range(sy):
                for i in range(sx):
                    auxX[j+sy-1][i+sx-1] = Bx[l+j][k+i]
                    auxY[j+sy-1][i+sx-1] = By[l+j][k+i]
                    auxZ[j+sy-1][i+sx-1] = Bz[l+j][k+i]
            for jj in range(2*sy-1):
                for ii in range(2*sx-1):
                    for j in range(sy):
                        for i in range(sx):
                            res_den += auxX[j+jj][i+ii] * Bx[l+j][k+i] + auxY[j+jj][i+ii] * By[l+j][k+i] + auxZ[j+jj][i+ii] * Bz[l+j][k+i]
                            res_num1 += ((float(ii)-float(sx)/2.0)**2.0+(float(jj)-float(sy)/2.0)**2.0)*(auxX[j+jj][i+ii] * Bx[l+j][k+i] + auxY[j+jj][i+ii] * By[l+j][k+i] + auxZ[j+jj][i+ii] * Bz[l+j][k+i])
                            res_num2 += (float(jj)-float(sy)/2.0)*(auxX[j+jj][i+ii] * Bx[l+j][k+i] + auxY[j+jj][i+ii] * By[l+j][k+i] + auxZ[j+jj][i+ii] * Bz[l+j][k+i])
            t1.append(res_num1/res_den)
            t2.append(res_num2/res_den)
        result1.append(t1)
        result2.append(t2)
    return result1, result2

def corrDectector2(Bx, By, Bz, sx, sy):
    lenX = len(Bx[0])
    lenY = len(Bx)
    result1 = []
    result2 = []
    for l in range(lenY-sy):
        t1 = []
        t2 = []
        for k in range(lenX-sx):
            res_num1 = 0.0
            res_num2 = 0.0
            res_den = 0.0
            for j in range(sy):
                for i in range(sx):
                    res_den += Bx[l][k] * Bx[l+j][k+i] + By[l][k] * By[l+j][k+i] + Bz[l][k] * Bz[l+j][k+i]
                    res_num1 += ((float(i)-float(sx)/2.0)**2.0+(float(j)-float(sy)/2.0)**2.0)*(Bx[l][k] * Bx[l+j][k+i] + By[l][k] * By[l+j][k+i] + Bz[l][k] * Bz[l+j][k+i])
                    #res_num1 += (float(i)-float(sx)/2.0)*(Bx[l][k] * Bx[l+j][k+i] + By[l][k] * By[l+j][k+i] + Bz[l][k] * Bz[l+j][k+i])
                    #res_num2 += (float(j)-float(sy)/2.0)*(Bx[l][k] * Bx[l+j][k+i] + By[l][k] * By[l+j][k+i] + Bz[l][k] * Bz[l+j][k+i])
            t1.append(res_num1/res_den)
            t2.append(res_num2/res_den)
        result1.append(t1)
        result2.append(t2)
    return result1, result2

def corrDectector3(Bz, sx, sy):
    lenX = len(Bz[0])
    lenY = len(Bz)
    result1 = []
    result2 = []
    for l in range(lenY-sy):
        t1 = []
        t2 = []
        for k in range(lenX-sx):
            res_num1 = 0.0
            res_num2 = 0.0
            res_den = 0.0
            for j in range(sy):
                for i in range(sx):
                    res_den += Bz[l][k] * Bz[l+j][k+i]
                    res_num1 += (float(i)-float(sx)/2.0)*(Bz[l][k] * Bz[l+j][k+i])
                    #res_num1 += ((float(i)-float(sx)/2.0)**2.0+(float(j)-float(sy)/2.0)**2.0)*(Bz[l][k] * Bz[l+j][k+i])
                    #res_num1 += (float(i)-float(sx)/2.0)*(Bx[l][k] * Bx[l+j][k+i] + By[l][k] * By[l+j][k+i] + Bz[l][k] * Bz[l+j][k+i])
                    #res_num2 += (float(j)-float(sy)/2.0)*(Bx[l][k] * Bx[l+j][k+i] + By[l][k] * By[l+j][k+i] + Bz[l][k] * Bz[l+j][k+i])
            t1.append(res_num1/res_den)
            t2.append(res_num2/res_den)
        result1.append(t1)
        result2.append(t2)
    return result1, result2

def corr(Bz, sx, sy):
    lenX = len(Bz[0])
    lenY = len(Bz)
    res = []
    for l in range(lenY-sy):
        t1 = []
        for k in range(lenX-sx):
            res_den = 0.0
            for j in range(sy):
                for i in range(sx):
                    res_den += Bz[l][k] * Bz[l+j][k+i]
            t1.append(res_den)
        res.append(t1)
    return 

def rms(Bz, sx, sy):
    lenX = len(Bz[0])
    lenY = len(Bz)
    res = []
    for l in range(lenY-sy):
        t1 = []
        for k in range(lenX-sx):
            res_den = 0.0
            for j in range(sy):
                for i in range(sx):
                    res_den += (Bz[l][k] - Bz[l+j][k+i])**2
            t1.append(res_den)
        res.append(t1)
    return res

def binarizationX(Bz, sx=3, sy=3):
    lenX = len(Bz[0])
    lenY = len(Bz)
    res = []
    for l in range(0, lenY):
        t1 = []
        for k in range(1, lenX):
            #res_den = (Bz[l+sy][k] + Bz[l][k+sx]+ Bz[l-sy][k] + Bz[l][k-sx]) / 4.0
            if abs(Bz[l][k-1]) < abs(Bz[l][k]):
                t1.append(1)
            else:
                t1.append(0)
        t1.append(0)
        res.append(t1)
    return np.array(res)

def laplasDetector(B, sx = 2, sy = 2):
    lenX = len(B[0])
    lenY = len(B)
    # 1  1  1
    # 1 -8  1
    # 1  1  1
    result = []
    for l in range(lenY-2*sy):
        t = []
        for k in range(lenX-2*sx):
            t.append(B[l][k] + B[l][k+sx] + B[l][k+2*sx] + B[l+sy][k] + B[l+sy][k+2*sx] + B[l+2*sy][k] + B[l+2*sy][k+sx] + B[l+2*sy][k+2*sx] - 8*B[l+sy][k+sx])
        result.append(t)
    return result

def binarizationY(Bz, sx=1, sy=1):
    lenX = len(Bz[0])
    lenY = len(Bz)
    res = []
    for l in range(sy, lenY-sy):
        t1 = []
        for k in range(sx, lenX-sx):
            res_den = (Bz[l+sy][k] + Bz[l][k+sx]+ Bz[l-sy][k] + Bz[l][k-sx]) / 4.0
            if abs(res_den)*1.5 < abs(Bz[l][k]):
                t1.append(1)
            else:
                t1.append(0)
        res.append(t1)
    return np.array(res)

def main(argv):
    X, Y, tan1, tan2, norm, vol, steps = readFile(argv[0])
    #Btest = np.zeros((100, 100))
    #X, Y = np.meshgrid(arange(10), arange(10))
    #Btest[5][5] = 1.0
    #Btest = rotate(Btest, 45, reshape=False)

    #res = spectrumDetector(np.array(norm), 20, 20, 5)
    #res1 = [[0.0 for i in range(len(norm[0]))] for j in range(len(norm))]
    #for j in range(len(res)):
    #    for i in range(len(res[0])):
    #        res1[j][i] = res[j][i]
    #saveFile("out1.txt", vol, steps, X, Y, tan1, tan2, res1)

    res = laplasDetector(norm, 10, 10)
    X, Y = np.meshgrid(np.arange(len(res[0])), np.arange(len(res)))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, res, rstride=1, cstride=1, cmap=cm.gist_rainbow)
    plt.show()
    
if __name__ == "__main__":
    main(sys.argv[1:])
