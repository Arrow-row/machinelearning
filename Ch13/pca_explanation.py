'''
Created on Jun 1, 2011

@author: Peter Harrington
'''
from numpy import *

def loadDataSet(fileName, delim='\t'): #从.txt文件中解析样本数据
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()] #line是文件中一行内容的字符串型数据
    #print('stringArr: ',type(stringArr),stringArr) #stringArr: <class 'list'> [['10.235186', '11.321997'], ['10.122339', '11.810993'], ['9.190236', '8.904943']]
    datArr = [list(map(float,line)) for line in stringArr]
    #print('datArr: ',type(datArr),datArr) #datArr: <class 'list'> [[10.235186, 11.321997], [10.122339, 11.810993], [9.190236, 8.904943]]
    return mat(datArr) #返回处理后的样本矩阵

def pca(dataMat, topNfeat=9999999): #本函数实现PCA算法。dataMat为待处理矩阵，topNfeat是用户指定主成分个数
    meanVals = mean(dataMat, axis=0) #Numpy中的mean()用于求均值，axis=0表示压缩行，对dataMat各列求均值
    meanRemoved = dataMat - meanVals #原数据矩阵dataMat中心化为meanRemoved，中心化后数据特征值的均值为0
    covMat = cov(meanRemoved, rowvar=0) #cov()函数计算meanRemoved的协方差矩阵covMat，rowvar=0表示矩阵的一列为一个特征变量
    #print('the type of covMat: ',type(covMat)) #the type of covMat:  <class 'numpy.ndarray'>
    eigVals,eigVects = linalg.eig(mat(covMat)) #linalg线性代数模块的eig()函数求矩阵covMat的特征值和特征向量
    eigValInd = argsort(eigVals)            #调用argsort()对特征值由小到大排序，返回排序后元素在原列表中的索引组成的矩阵   sort, sort goes smallest to largest
    #print('eigValInd before election: ',eigValInd)
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #列表eigValInd尾部开始取topNfeat个元素，对应前topNfeat个最大的特征值    cut off unwanted dimensions
    #print('eigValInd after election: ',eigValInd)
    #print('eigVects before election: ',eigVects)
    redEigVects = eigVects[:,eigValInd]       #选择前topNfeat个最大的特征值对应的特征向量作为主成分，redEigVects是主成分向量组成的矩阵   reorganize eig vects largest to smallest
    #print('eigVects after election: ',shape(redEigVects),redEigVects)
    lowDDataMat = meanRemoved * redEigVects   #原矩阵降维后得到数据矩阵lowDDataMat，是中心化形式    transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals #原始数据在主成分方向上的投影，非中心化
    #print(type(lowDDataMat),type(reconMat)) #<class 'numpy.matrixlib.defmatrix.matrix'> <class 'numpy.matrixlib.defmatrix.matrix'>
    return lowDDataMat, reconMat

    #dataMat: mxn，原数据矩阵 
    #meanRemoved: mxn，dataMat中心化 
    #covMat: nxn，meanRemoved求协方差矩阵 
    #eigVects: nxn，covMat特征向量矩阵 
    #redEigVects: nxk，eigVects前k个最大特征值对应特征向量构成
    #lowDDataMat: mxk，dataMat降维后的矩阵,是中心化后的形式，中心在原点
    #reconMat: mxn，原始数据在主成分方向上的投影，非中心化

def replaceNanWithMean(): #本函数功能：将数据集中缺失的样本特征值用该特征的均值填充
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1] #获取样本特征数量（矩阵列数）
    for i in range(numFeat): #遍历所有特征
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #计算datMat中第i列的均值meanVal(去掉无值元素、0值元素)。isnan(datMat[:,i].A):判断datMat第i列的各元素是否是空值NaN,是返回True,否返回False; nonzero(~isnan(datMat[:,i].A))[0]:获取第i列中有值的非0行的行索引  values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #将第i列的空值位置(NaN)以meanVal填充      set NaN values to mean
    return datMat #返回填充处理后的矩阵
