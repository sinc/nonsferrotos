from skimage.feature import peak_local_max
import numpy as np
from . import detectors as dtc
from . import vec3Field as v3f


def detectorAnalyseData(field,detectorType,winSize = 2,distanceBetweenPoints = 3, typeOfResult = 'indicies'):
    """find intersting for analyse points - maxes of detector"""
    #calc crack-detector
    resCrack = dtc.gradDetector(field,winSize,detectorType)
    #find maxes of crack-detector
    maxes = peak_local_max(np.array(resCrack.Bz),min_distance = distanceBetweenPoints , indices =True)
    #sort maxes in 
    maxes = sorted(maxes,key = lambda item: resCrack.Bz[item[0]][item[1]],reverse = True)
    #find inicies of maxes in source field
    if(typeOfResult == 'indicies'):
        print(maxes[0])
        return resCrack,[v3f.nearIndicies(field.vol,resCrack.X,resCrack.Y,field.steps,[point[1],point[0]]) for point in maxes]
    else:
        if(typeOfResult =='coords'):
            return resCrack,[v3f.nearCoordinates(field.vol,resCrack.X,resCrack.Y,field.steps,[point[1],point[0]]) for point in maxes]
        else:
            return resCrack,[]
def markingCracks(detector):
    """It will be used for painting cracks"""
    pass