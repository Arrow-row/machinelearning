'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator

def createDataSet(): #构建简单数据集
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    #change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):  #计算给定数据集的香农熵
    numEntries = len(dataSet)  #获取数据集样本总数
    labelCounts = {}  #类别标签字典
    for featVec in dataSet:  #统计样本的各类标签出现次数
        currentLabel = featVec[-1]  #currentLabel记录当前样本的类别标签，即当前样本向量的最后一个元素
        if currentLabel not in labelCounts.keys(): #当前样本的类别标签未记录在字典labelCounts的中
            labelCounts[currentLabel] = 0 #向字典中添加key-value,key=currentLabel,value=0
        labelCounts[currentLabel] += 1 #当前标签出现的次数+1
    shannonEnt = 0.0 #初始化Ent=0
    for key in labelCounts: #遍历字典labelCounts中key值，即不同类型的标签
        prob = float(labelCounts[key])/numEntries #计算各类型标签在数据集中出现的概率
        shannonEnt -= prob * log(prob, 2) #根据香农熵公式计算给定数据集的香农熵。log是math模块中的函数,log(prob,2)：返回以2为低prob的对数
    return shannonEnt #返回本数据集的香农熵

def splitDataSet(dataSet, axis, value):  #按照给定特征axis及其值value划分数据集dataSet
    retDataSet = [] #创建新的list对象retDataSet存放筛选出的数据。因为本函数代码在同一数据集上被调用多次，为了不修改原始数据集，创建一个新的列表对象
    for featVec in dataSet:  #遍历数据集中样本
        if featVec[axis] == value:  #选择特征axis取值为value的样本（抽取符合特征的数据）
            reducedFeatVec = featVec[:axis]     #本行和下面一行去掉样本向量featVec中的特征axis，返回reducedFeatVec向量
            reducedFeatVec.extend(featVec[axis+1:]) 
            retDataSet.append(reducedFeatVec)  #将reducedFeatVec向量追加到数据集retDataSet
    return retDataSet  #返回本次筛选出的数据集

def chooseBestFeatureToSplit(dataSet):  #通过计算不同特征值划分后数据子集的香农熵，选择最好的数据划分方式
    numFeatures = len(dataSet[0]) - 1   #计算样本向量中样本个数，向量的最后一个元素是类别标签
    baseEntropy = calcShannonEnt(dataSet)  #计算整个数据集的香农熵
    bestInfoGain = 0.0; bestFeature = -1   
    for i in range(numFeatures):        #遍历所有特征
        featList = [example[i] for example in dataSet]  #取数据集dataSet中所有样本的第i个特征,存于列表featList,列表中元素可重复。 
        uniqueVals = set(featList)      #对featList数据去重得到集合uniqueVals
        newEntropy = 0.0  #按特征i划分后的数据集香农熵
        for value in uniqueVals:  #依次获取第i个特征的特征值value
            subDataSet = splitDataSet(dataSet, i, value)  #调用数据集划分函数,根据特征i的特征值value划分dataSet,返回划分出的子集subDataSet
            prob = len(subDataSet)/float(len(dataSet))  #当前特征值样本在总样本中的占比
            newEntropy += prob * calcShannonEnt(subDataSet)  #当前特征值子集香农熵乘以子集占比，再求和，得到按特征i划分后的数据集香农熵newEntropy。越有序的划分，newEntropy越小
        infoGain = baseEntropy - newEntropy #计算按当前特征划分后，信息增益=原始信息熵-新信息熵。新信息熵越小，信息增益越大
        if (infoGain > bestInfoGain):       #使得信息增益最大的特征划分就是最佳数据划分方式
            bestInfoGain = infoGain         
            bestFeature = i                 
    return bestFeature                      #返回最佳划分对应的特征下标

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree)[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)
