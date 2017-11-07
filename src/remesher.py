import numpy as np
from scipy.interpolate import RectBivariateSpline

from . import vec3Field as v3f    

def recalcMagnFieldinXY(field,sensorCoords):
    """#we recalc tan magnetic field to surface of norm magnetic field
    #coordinates of sensors relatively to Bz sensor
    #sensorCoords = [[-1.7,0,-1.7],[-.3,-1.3,-1.7]]
    """
    #recalculating x-component - the first sensor
    y_grid = np.arange((field.vol[1]-sensorCoords[0][1]),(field.vol[1]+field.vol[4]-sensorCoords[0][1]+field.steps[1]),field.steps[1])
    x_grid = np.arange((field.vol[0]-sensorCoords[0][0]),(field.vol[0]+field.vol[3]-sensorCoords[0][0]+field.steps[0]),field.steps[0])
    approx = RectBivariateSpline(y_grid, x_grid, field.Bx)
    Xnew,Ynew = np.meshgrid(x_grid,y_grid)
    data_X = [[approx(x,y) for x in x_grid] for y in y_grid]
    
    #recalculating y-component - the second sensor
    y_grid = np.arange((field.vol[1]-sensorCoords[1][1]),(field.vol[1]+field.vol[4]-sensorCoords[1][1]+field.steps[1]),field.steps[1])
    x_grid = np.arange((field.vol[0]-sensorCoords[1][0]),(field.vol[0]+field.vol[3]-sensorCoords[1][0]+field.steps[0]),field.steps[0])
    approx = RectBivariateSpline(y_grid, x_grid, field.Bx)
    Xnew,Ynew = np.meshgrid(x_grid,y_grid)
    data_Y = [[approx(x,y) for x in x_grid] for y in y_grid]
    
    #we return result in grid of the source field as a grid of the third z-component sensor
    return v3f.vec3Field(field.X,field.Y,data_X,data_Y,field.Bz,field.vol,field.steps)
    #return X1,Y1,tan1_r,tan2_r,norm,[1*steps[1]+vol[1],1*steps[0]+vol[0],vol[2],(len(norm[0])-2)*(steps[1]),(len(norm)-2)*(steps[0]),0],steps
