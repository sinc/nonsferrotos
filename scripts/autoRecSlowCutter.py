#!/usr/bin/env pytnon
#
import sys
from os import listdir
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

sys.path.append('..\\..\\')
import nonsferrotos.src.vec3Field as v3f
import nonsferrotos.src.cutters as ctrs
import nonsferrotos.src.remesher as rmsh
def main(argv):
    i=0
    sensors = [[-1.7,0,-1.7],[-.3,-1.3,-1.7]]
    #secondwd argv argument allow to set output path
    #if it passed, output path is equal to the input path
    if(len(argv)>1):
        newFName = lambda fName: argv[1]+fName+'t'
    else:
        newFName = lambda fName: fName+'t'
    for filename in listdir(argv[0]):
        data = v3f.readFile(argv[0]+'\\'+filename)
        print(filename)
        print('source:',data)
        data = ctrs.autoRecRegionCutter(data)
        print('RecCutted:',data)
        data = ctrs.slowCutter(data)
        print('SlowCutted:',data)
        v3f.saveFile(data,newFName(filename))
        data = rmsh.recalcMagnFieldinXY(data,sensors)
        print('RecCutted:',data)
        i+=1
    print('ok',i,' files.')
if __name__ == "__main__":
    main(sys.argv[1:])

