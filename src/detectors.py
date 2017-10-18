import numpy as np
from . import vec3Field as v3f

#integrals are token by average window - winSize
#type - means detector type used for processing
def gradDetector(field,winSize,type):
    grad = np.gradient(field.Bz)
    #calculate detector by formula Integrate[dBz/dx*dBz/dy]
    if(type == 'doubleGrad'):
        data_grad = grad[0]*grad[1]
        dataRet = [[sum(data_grad[i:i+winSize,j:j+winSize].ravel()) for j in range(len(data_grad[0])-winSize)] for i in range(len(data_grad)-winSize)]
    else:
        data_X = grad[0]*field.Bx
        data_Y = grad[1]*field.By
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
    volNew = [field.vol[0]+field.steps[0]+winSize/2.0,field.vol[1]+field.steps[1]+winSize/2.0,field.vol[2],field.vol[3]-(winSize-1)*field.steps[0],field.vol[4]-(winSize-1)*field.steps[1],field.vol[5]]
    X1, Y1  = np.meshgrid(
        np.arange(volNew[0],volNew[0]+volNew[3],field.steps[0]),
        np.arange(volNew[1],volNew[1]+volNew[4],field.steps[1]),
        )
    return v3f.vec3Field(X1,Y1,[],[],dataRet,volNew,field.steps)

