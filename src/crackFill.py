import numpy as np
def pointsToAnalyseReturn(logicMassive,startPoint,label):
    predv = [[startPoint[0],startPoint[1]+1],[startPoint[0]+1,startPoint[1]],[startPoint[0],startPoint[1]+1],[startPoint[0]+1,startPoint[1]+1],
           [startPoint[0],startPoint[1]-1],[startPoint[0]-1,startPoint[1]],[startPoint[0],startPoint[1]-1],[startPoint[0]-1,startPoint[1]-1],
           [startPoint[0]+1,startPoint[1]-1],[startPoint[0]-1,startPoint[1]+1]]
    ret = []
    for point in predv:
        #print('p[',point[0],',',point[1],']')
        if(point[0]>=0 and point[0]<len(logicMassive[0]) and point[1]>=0 and point[1]<len(logicMassive)):
            if(logicMassive[point[1]][point[0]]!=label):
                ret.append(point)
    return ret
def pointsToAnalyseLineReturn(logicMassive,startPoint,label):
    predv =[[startPoint[0]+1,startPoint[1]+1],[startPoint[0],startPoint[1]+1],[startPoint[0]-1,startPoint[1]+1],[startPoint[0]-1,startPoint[1]],
            [startPoint[0]-1,startPoint[1]-1],[startPoint[0],startPoint[1]-1],[startPoint[0]+1,startPoint[1]-1],
            [startPoint[0]+1,startPoint[1]],[startPoint[0]+1,startPoint[1]+1],[startPoint[0],startPoint[1]+1]]
    ret = []
    for it in range(1,len(predv)-2):
        #print('p[',point[0],',',point[1],']')
        point = predv[it]
        if(predv[it][0]>=0 and predv[it][0]<len(logicMassive[0]) and predv[it][1]>=0 and predv[it][1]<len(logicMassive) and predv[it-1][0]>=0 and predv[it-1][0]<len(logicMassive[0]) and predv[it-1][1]>=0 and predv[it-1][1]<len(logicMassive) and predv[it+1][0]>=0 and predv[it+1][0]<len(logicMassive[0]) and predv[it+1][1]>=0 and predv[it+1][1]<len(logicMassive)):
            if(logicMassive[point[1]][point[0]]!=label and logicMassive[predv[it-1][1]][predv[it-1][0]]!=label and logicMassive[predv[it+1][1]][predv[it+1][0]]!=label):
                ret.append(point)
    return ret
def pointsToAnalyseLinePredictionReturn(logicMassive,startPoint,label):
    predv =[[startPoint[0],startPoint[1]+1],[startPoint[0]-1,startPoint[1]+1],[startPoint[0]-1,startPoint[1]],
            [startPoint[0]-1,startPoint[1]-1],[startPoint[0],startPoint[1]-1],[startPoint[0]+1,startPoint[1]-1],
            [startPoint[0]+1,startPoint[1]],[startPoint[0]+1,startPoint[1]+1]]
    ret = []
    for it in range(len(predv)):
        if(predv[it][0]>=0 and predv[it][0]<len(logicMassive[0]) and predv[it][1]>=0 and predv[it][1]<len(logicMassive)):
            if(logicMassive[predv[it][1]][predv[it][0]]==label):
                predv = [predv[it+3] if (it+3)<len(predv) else predv[it+3-len(predv)],
                         predv[it+4] if (it+4)<len(predv) else predv[it+4-len(predv)],
                         predv[it+5] if (it+5)<len(predv) else predv[it+5-len(predv)]]
                break
    for it in range(len(predv)):
        if(predv[it][0]>=0 and predv[it][0]<len(logicMassive[0]) and predv[it][1]>=0 and predv[it][1]<len(logicMassive)):
            if(logicMassive[predv[it][1]][predv[it][0]]==label):
                return []
            else:
                ret.append(predv[it])
    return ret
def lineGeneration(points,data):
    ret = np.zeros((len(data),len(data[0])))
    pointAndValues = [[point[1],point[0],data[point[0]][point[1]]] for point in points]
    print(pointAndValues)
    pointAndValues= sorted(pointAndValues,key = lambda item: item[2],reverse = True)
    print(pointAndValues)
    for ip in range(0,len(pointAndValues)):
        startx = int(pointAndValues[ip][0])
        starty = int(pointAndValues[ip][1])
        startF = pointAndValues[ip][2]
        kolvo = 20
        i = 0
        while True:
            #print('data:',len(data[0]),len(data))
            #print('ret:',len(ret[0]),len(ret))
            #print(startx,starty)
            ret[starty][startx] = ip+1
            #analysePoints = pointsToAnalyseLineReturn(ret,[startx,starty],ip+1)
            analysePoints = pointsToAnalyseLinePredictionReturn(ret,[startx,starty],ip+1)
            values = [data[point[1]][point[0]] for point in analysePoints]
            #print('valCount=',len(values))
            if(len(values)!=0):
                maxInd = np.argmax(values)
                minInd = np.argmin(values)
                #avgVal = np.average(values)
                #print('val',values)
                if(values[maxInd] > startF*0.5 and values[maxInd] < startF*1.5):
                #if(i > 20):
                    startx = analysePoints[maxInd][0]
                    starty = analysePoints[maxInd][1]
                    #print('newPoint:',startx,starty)
                    i+=1
                else:
                    break
            else:
                break
    return ret
def lineGeneration2(points,data):
    ret = np.zeros((len(data),len(data[0])))
    labels = lineGeneration(points,data)
    pointAndValues = [[point[1],point[0],data[point[0]][point[1]]] for point in points]
    print(pointAndValues)
    pointAndValues= sorted(pointAndValues,key = lambda item: item[2])
    print(pointAndValues)
    for ip in range(0,len(pointAndValues)):
        startx = int(pointAndValues[ip][0])
        starty = int(pointAndValues[ip][1])
        startF = pointAndValues[ip][2]
        kolvo = 20
        i = 0
        while True:
            #print('data:',len(data[0]),len(data))
            #print('ret:',len(ret[0]),len(ret))
            #print(startx,starty)
            ret[starty][startx] = ip+1
            analysePoints = pointsToAnalyseReturn(ret,[startx,starty],ip+1)
            values = [data[point[1]][point[0]] for point in analysePoints]
            #print('valCount=',len(values))
            if(len(values)!=0):
                maxInd = np.argmax(values)
                minInd = np.argmin(values)
                #avgVal = np.average(values)
                #print('val',values)
                if(values[maxInd] > startF*0.5 and values[maxInd] < startF*1.5):
                #if(i > 20):
                    startx = analysePoints[maxInd][0]
                    starty = analysePoints[maxInd][1]
                    #print('newPoint:',startx,starty)
                    i+=1
                else:
                    break
            else:
                break
    return ret
