import numpy 

#calculate field of simple 
def dipole(x, y, z, dx, dy, dz, mx, my, mz):
    R = (x - dx)**2 + (y - dy)**2 + (z - dz)**2
    return (3.0*(x - dx) * ((x - dx)*mx + (y - dy)*my + (z - dz)*mz) / R**2.5 - mx/R**1.5,
            3.0*(y - dy) * ((x - dx)*mx + (y - dy)*my + (z - dz)*mz) / R**2.5 - my/R**1.5,
            3.0*(z - dz) * ((x - dx)*mx + (y - dy)*my + (z - dz)*mz) / R**2.5 - mz/R**1.5)

#calculate field caused by crack from array of coordinates and magntization of crack parts
def crack(x,y,z,coordinates,magnetization):
    ret = numpy.array([0.0]*3)
    for it in range(len(coordinates)):
        ret+=numpy.array(dipole(x,y,z,coordinates[it][0],coordinates[it][1],coordinates[it][2],magnetization[it][0],magnetization[it][1],magnetization[it][2]))
    return ret

#generator of crack parts coordinates and magntization 
def crackGenerator(funcCoord, funcMagn,crackLen = 30, paramBouns = [0,1]):
    coordinates = []
    magnetization = []
    for t in numpy.arange(paramBouns[0],paramBouns[1],(paramBouns[1]-paramBouns[0])/crackLen):
        coordinates.append(funcCoord(t))
        magnetization.append(funcMagn(t))
    return coordinates,magnetization

#generates one random crack in volume vol
def randomCrackExampleLinearModel(vol):
    sizeMax = (vol[3]/5,vol[4]/5,vol[5]/5)
    coordParams = numpy.random.rand(3,2)
    return crackGenerator(lambda t:(coordParams[0][0]*vol[3]+vol[0]+t*coordParams[0][1]*sizeMax[0],
                             coordParams[1][0]*vol[4]+vol[1]+t*coordParams[1][1]*sizeMax[1],
                             coordParams[2][0]*vol[5]+vol[2]+t*coordParams[2][1]*sizeMax[2]),
                   lambda t: (0,0,10+numpy.random.rand()*t))