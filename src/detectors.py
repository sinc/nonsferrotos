import numpy as np
from . import vec3Field as v3f

#integrals are token by average window - winSize
#type - means detector type used for processing
def gradDetector(field,winSize = 4,type = 'crack',absVal = 'False'):
    grad = np.gradient(field.Bz)
    #calculate detector by formula Integrate[dBz/dx*dBz/dy]
    if(type == 'doubleGrad'):
        if(absVal):
            data_grad = np.abs(grad[0])*np.abs(grad[1])
        else:
            data_grad = (grad[0])*(grad[1])
        dataRet = [[sum(data_grad[i:i+winSize,j:j+winSize].ravel()) for j in range(len(data_grad[0])-winSize)] for i in range(len(data_grad)-winSize)]
    else:
        if(absVal):
            data_X, data_Y = np.abs(grad[1]*field.Bx),np.abs(grad[0]*field.By)
        else:
            data_X, data_Y = grad[1]*field.Bx,grad[0]*field.By
        #calculate detector by formula Integrate[dBz/dx*Bx]*Integrate[dBz/dy*By] 
        if(type == 'dipole'):    
            dataRet = [[sum(data_X[i:i+winSize,j:j+winSize].ravel())*sum(data_Y[i:i+winSize,j:j+winSize].ravel()) for j in range(len(data_X[0])-winSize)] for i in range(len(data_X)-winSize)]
        else: 
            #calculate detector by formula Integrate[dBz/dx*Bx]+Integrate[dBz/dy*By] 
            if(type == 'crack'):
                dataRet = [[sum(data_X[i:i+winSize,j:j+winSize].ravel())+sum(data_Y[i:i+winSize,j:j+winSize].ravel()) for j in range(len(data_X[0])-winSize)] for i in range(len(data_X)-winSize)]
            else:
                print('unknow type of detector')
                return None
    #calculating new grid
    volNew = [field.vol[0]+field.steps[0]+winSize/2.0,field.vol[1]+field.steps[1]+winSize/2.0,field.vol[2],field.vol[3]-(winSize)*field.steps[0],field.vol[4]-(winSize)*field.steps[1],field.vol[5]]
    X1, Y1  = np.meshgrid(
        np.linspace(volNew[0],volNew[0]+volNew[3],num =len(field.X[0])-winSize),
        np.linspace(volNew[1],volNew[1]+volNew[4],num =len(field.X)-winSize),
        )
    return v3f.vec3Field(X1,Y1,[],[],dataRet,volNew,field.steps)

