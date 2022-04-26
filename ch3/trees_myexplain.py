'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator

def createDataSet(): #构建简单数据集
    dataSet = [[1, 1, 'yes'],  #列表dataSet中元素仍为列表，元素的前2列为样本特征的取值，第3列为数据划分的最终类别
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']    #labels为样本特征列表
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

def splitDataSet(dataSet, axis, value):  #按照给定特征及其值value划分数据集dataSet,axis是给定特征在特征向量labels上的索引
    retDataSet = [] #创建新的list对象retDataSet存放筛选出的数据。因为本函数代码在同一数据集上被调用多次，为了不修改原始数据集，创建一个新的列表对象
    for featVec in dataSet:  #遍历数据集中样本
        if featVec[axis] == value:  #选择特征axis取值为value的样本（抽取符合特征的数据）
            reducedFeatVec = featVec[:axis]     #本行和下面一行去掉样本向量featVec中的特征axis，返回reducedFeatVec向量
            reducedFeatVec.extend(featVec[axis+1:]) 
            retDataSet.append(reducedFeatVec)  #将reducedFeatVec向量追加到数据集retDataSet
    return retDataSet  #返回本次筛选出的数据集

def chooseBestFeatureToSplit(dataSet):  #通过计算不同特征值划分后数据子集的香农熵，选择最好的划分数据集的特征
    numFeatures = len(dataSet[0]) - 1   #计算样本向量中特征个数，向量的最后一个元素是类别标签。dataSet类型是列表list,其中元素也是list,dataSet[0]表示取第一条样本
    baseEntropy = calcShannonEnt(dataSet)  #计算整个数据集的原始香农熵
    bestInfoGain = 0.0; bestFeature = -1   #初始化最佳增益和最佳特征索引
    for i in range(numFeatures):        #遍历所有特征
        featList = [example[i] for example in dataSet]  #使用for循环,依次取出数据集dataSet中所有样本第i个特征的特征值,存于列表featList,列表中元素可重复
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
    classList = [example[-1] for example in dataSet]  #取出数据集中每条样本的类标签,组成类型标签列表。dataSet和classlist均为列表list
    if classList.count(classList[0]) == len(classList):  #第一个递归出口：若数据集中类型标签全部相同,停止划分。count()函数统计列表中元素出现次数
        return classList[0]  #返回第一个类标签
    if len(dataSet[0]) == 1: #第二个递归出口：若数据集中的向量只剩下类标签元素,无法再划分
        return majorityCnt(classList) #调用majorityCnt()，返回类标签向量中出现次数最多的标签
    bestFeat = chooseBestFeatureToSplit(dataSet)  #调用chooseBestFeatureToSplit()，计算出数据集最佳划分特征索引bestFeat
    bestFeatLabel = labels[bestFeat]  #获取本次划分的特征bestFeatLabel，labels是数据集的特征向量
    myTree = {bestFeatLabel:{}}  #字典myTree存储决策树信息,每次选出的最佳特征作为key(即根节点)存入myTree
    del(labels[bestFeat])  #每次划分完成后将bestFeat从特征向量labels中删除
    featValues = [example[bestFeat] for example in dataSet]  #取出数据集中每条样本的bestFeat特征值,组成特征值列表featValues
    uniqueVals = set(featValues) #featValues列表元素去重
    for value in uniqueVals:  #循环调用splitDataSet(),根据特征bestFeatLabel的值value划分dataSet
        subLabels = labels[:]       #拷贝当前特征向量(函数参数是列表类型时,参数是按照引用方式传递的,为保证每次调用函数createTree()时不改变原始列表的内容，使用新变量subLabels代替原始列表)        copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)  #使用splitDataSet()划分后返回的数据集递归创建决策树,createTree()返回的类标签作为决策时字典值value存入myTree
    return myTree  #返回决策树字典信息

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

