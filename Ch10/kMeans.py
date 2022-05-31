'''
Created on Feb 16, 2011
k Means Clustering for Ch10 of Machine Learning in Action
@author: Peter Harrington
'''
from numpy import *

def loadDataSet(fileName):     #从文件中解析数据，转化为flout类型，存入矩阵dataMat
    dataMat = []                
    fr = open(fileName)
    for line in fr.readlines():
        #print(type(line),len(line),line[0],line[1],line)  #<class 'str'> 20 - 4 -4.905566  -2.911070
        curLine = line.strip().split('\t')
        #print(type(curLine),len(curLine),curLine[0],curLine[1],curLine) #<class 'list'> 2 -4.905566 -2.911070 ['-4.905566', '-2.911070']
        fltLine = list(map(float,curLine)) #列表字符串元素转换为float类型 
        #print(type(fltLine),len(fltLine),fltLine[0],fltLine[1],fltLine) #<class 'list'> 2 -4.905566 -2.91107 [-4.905566, -2.91107]
        dataMat.append(fltLine) #数据行以列表存入dataMat
    return dataMat #返回数据矩阵

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #计算两向量的欧式距离，各分量距离的平方和再开方   la.norm(vecA-vecB)

def randCent(dataSet, k): #为数据集构建k个随机质心
    n = shape(dataSet)[1] #n为样本特征个数
    centroids = mat(zeros((k,n))) #初始化k个质心向量组成的kxn矩阵   create centroid mat
    for j in range(n): #遍历n个特征    create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j]) #取第j个特征的最小值minJ
        rangeJ = float(max(dataSet[:,j]) - minJ) #求j的特征值跨度范围rangeJ
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1)) #rand函数根据给定维度(k,1)生成[0,1)之间的数据，包含0，不包含1；生成的数据类型为numpy.ndarray，可直接与数值相加，生成特征j的介于最大值、最小值直接的k个特征值，并形成kx1向量;最后转换为mat类存入centroids
    return centroids #返回k个随机质心组成的矩阵centroids
    
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent): #kMeans算法实现。dataSet:待处理数据集,k:簇的数目,distMeas:样本距离计算函数的引用,createCent:随机质心构建函数的引用
    m = shape(dataSet)[0] #m为数据集dataSet中样本个数
    clusterAssment = mat(zeros((m,2)))#mx2的矩阵，用于辅助数据点归类。第1列为数据点类别，第2列是数据点到最近质心的距离的平方     create mat to assign data points 
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k) #调用createCent(),构建随机质心矩阵centroids
    clusterChanged = True #clusterChanged：迭代停止标识，初始化为True
    while clusterChanged: #若簇中数据点分配结果不再改变(clusterChanged = False)，表示聚类完成，停止迭代,否则继续
        clusterChanged = False
        for i in range(m):#遍历数据集中样本点（为找到距离每个点最近的质心）        for each data point assign it to the closest centroid
            minDist = inf #最小距离初始化为无穷
            minIndex = -1 #最近质心点索引初始化为-1
            for j in range(k): #遍历质心，调用distMeas()计算样本i(dataSet[i,:])到每个质心j(centroids[j,:])之间的距离
                distJI = distMeas(centroids[j,:],dataSet[i,:]) 
                if distJI < minDist: #如果本次计算得到的距离小于前次，则更新样本质心最小距离minDist和最近质心索引minIndex
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True #样本点i的已有类别与当前计算出的类别不相等，则clusterChanged标识为True，表示需要继续迭代。只要数据集中有一个样本点的类别改变，迭代就会继续
            clusterAssment[i,:] = minIndex,minDist**2 #将当前计算出的最近质心索引、对应的最小距离平方记录于归类矩阵clusterAssment
        print(centroids) #打印当前质心矩阵
        for cent in range(k):#数据集中样本点的类别全部更新之后，用新的聚类情况重新计算质心矩阵 recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#通过数组过滤来获取类别cent中的所有样本点，ptsInClust是类别cent中的样本形成的矩阵     get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #mean(ptsInClust, axis=0)：样本矩阵ptsInClust中的向量按列求均值，得到的值作为质心cent的新坐标。axis = 0表示沿矩阵的列方向进行均值计算       assign centroid to mean 
    return centroids, clusterAssment #返回聚类完成后的质心矩阵和样本点分类结果

def biKmeans(dataSet, k, distMeas=distEclud): #二分kMeans算法
    m = shape(dataSet)[0] #样本点个数
    clusterAssment = mat(zeros((m,2)))  #mx2的矩阵，用于辅助数据点归类。第1列为数据点类别，第2列是数据点到最近质心的距离的平方
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0] #create a list with one centroid
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is: ',bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment

import urllib
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.parse.urlencode(params)
    yahooApi = apiStem + url_params      #print url_params
    print(yahooApi)
    c=urllib.request.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print("error fetching")
        sleep(1)
    fw.close()
    
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()
