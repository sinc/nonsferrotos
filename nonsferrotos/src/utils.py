import numpy as np

def sinWindow(lenX, lenY):
    return np.array([[(np.sin(np.pi * i/(lenX - 1))**2)*(np.sin(np.pi * j/(lenY - 1))**2) for i in range(lenX)] for j in range(lenY)])