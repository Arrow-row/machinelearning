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
    numFeatures = len(dataSet[0]) - 1   #计算样本向量中特征个数，向量的最后一个元素是类别标签。dataSet类型是列表list,其中元素也是list,dataSet[0]表示取第一条样本
    baseEntropy = calcShannonEnt(dataSet)  #计算整个数据集的原始香农熵
    bestInfoGain = 0.0; bestFeature = -1   #初始化最佳增益和最佳特征索引
    for i in range(numFeatures):        #遍历所有特征
        featList = [example[i] for example in dataSet]  #使用for循环,依次取出数据集dataSet中所有样本的第i个特征,存于列表featList,列表中元素可重复
        uniqueVals = set(featList)      #对featList数据去重得到集合uniqueVals，特征值value取值唯一
        newEntropy = 0.0  #按特征i划分后的所有数据子集香农熵之和
        for value in uniqueVals:  #获取当前特征的所有唯一特征值value
            subDataSet = splitDataSet(dataSet, i, value)  #调用划分函数splitDataSet,根据特征i的特征值value划分dataSet,返回划分出的子集subDataSet
            prob = len(subDataSet)/float(len(dataSet))  #包含当前特征值的样本在总样本中的占比
            newEntropy += prob * calcShannonEnt(subDataSet)  #当前特征值子集香农熵乘以子集占比，再求和，得到按特征i的各value划分后的数据集香农熵newEntropy。越有序的划分，newEntropy越小
        infoGain = baseEntropy - newEntropy #计算按当前特征划分后，信息增益=原始信息熵-新信息熵。新数据集的信息熵越小，信息增益infoGain越大
        if (infoGain > bestInfoGain):       #使得信息增益最大的特征划分就是最佳数据划分方式
            bestInfoGain = infoGain         
            bestFeature = i                 
    return bestFeature                      #返回最佳划分对应的特征索引

def majorityCnt(classList):  #本函数作用：当叶子节点中类标签不唯一时，通过多数表决方式决定该叶子节点的类型标签。classList为叶子节点数据集中的样本标签列表,其中的元素数据值可重复
    classCount={}  #声明字典变量classCount，用于记录叶子节点数据集中不同label出现的次数
    for vote in classList:  #依次获取classList中的标签
        if vote not in classCount.keys(): 
            classCount[vote] = 0  #当前标签不在字典classCount中,将其加入,key=label,value=0
        classCount[vote] += 1   #统计当前标签出现次数
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  #将classCount中数据按value值降序排列,返回重新排序后的列表。items()获取字典中的所有key-value对,返回可迭代对象dict_items;key表示取自可迭代对象、(进行一定运算后)用于进行比较的元素,这里通过itemgetter(1)取字典元素中的value;reverse=True将排序规则设置为降序
    return sortedClassCount[0][0]  #取出现次数最多的标签      sortedClassCount   [('H', 9), ('B', 6), ('A', 4)]

def createTree(dataSet, labels):  #创建决策树
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

