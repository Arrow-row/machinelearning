'''
Created on Feb 4, 2011
Tree-Based Regression Methods
@author: Peter Harrington
'''
from numpy import *

def loadDataSet(fileName):      #从文件中解析数据 general function to parse tab -delimited floats
    dataMat = []                #样本数据矩阵 assume last column is target value
    fr = open(fileName) #以只读形式打开ex0.txt文件，返回文件对象fr
    for line in fr.readlines(): #依次遍历文件中每一行。fr.readlines()读取.txt文件中所有行并以每一行内容作为元素返回一个列表
        curLine = line.strip().split('\t') #以'\t'对每行字符串进行切片，strip()去除首尾空格，返回分割后的字符串元素组成的列表给curLine
        fltLine = list(map(float,curLine)) #将每行内容保存为一组浮点数fltLine  map all elements to float()
        dataMat.append(fltLine) #将转换后的fltLine存入dataMat
    return dataMat #返回数据矩阵dataMat

def binSplitDataSet(dataSet, feature, value): #按给定特征和特征值，将数据集dataSet切分为两个子集，即数组过滤。dataSet：数据集，feature：待划分的特征，用于指定矩阵列号；value：待划分特征的某个值
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:] #feature表示第i列，dataSet[:,feature] > value返回dataSet第i列中大于value的元素，nonzero()[]返回数组中值不为0的元素的第0轴的下标。mat0为从dataSet中取出的样本点，满足：feature值>value、feature值非0
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:] #mat1中的样本点满足：feature值<=value、feature值非0
    return mat0,mat1 #返回切分后的两个子集

def regLeaf(dataSet): #此函数返回叶节点代表的标签值，取叶结点中所有样本点的均值     returns the value used for each leaf
    return mean(dataSet[:,-1]) #mean()用于取均值

def regErr(dataSet): #误差估计函数 
    return var(dataSet[:,-1]) * shape(dataSet)[0] #返回值为总方差，采用均方差乘以数据集中样本的个数得到总方差

def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)): #此函数在给定的误差计算方法下，找到数据集的最佳二元划分方式，返回达到最佳划分的特征和特征值。输入参数：数据集dataSet，建立叶节点的函数leafType，误差计算函数errType，包含树构建所需其他参数的元组ops
    tolS = ops[0]; tolN = ops[1]  #控制函数停止时机。tolS是容许的误差下降值，tolN是切分的最少样本数
    #if all the target variables are the same value: quit and return value
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #退出条件1   
        return None, leafType(dataSet) #leafType是对创建叶节点的函数的引用
    m,n = shape(dataSet)
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS: 
        return None, leafType(dataSet) #退出条件2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #退出条件3
        return None, leafType(dataSet)
    return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)): #树构建函数。输入参数：数据集dataSet，建立叶节点的函数leafType，误差计算函数errType，包含树构建所需其他参数的元组ops      assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops) #调用chooseBestSplit()选择最好的划分，返回选出的最好划分特征和特征值      choose the best split
    if feat == None: return val #满足停止条件feat == None时，表示当前节点划分结束，退出本层递归，返回叶节点值     if the splitting hit a stop condition return val
    retTree = {} #不满足退出条件，则创建字典存储回归树新一层的划分节点
    retTree['spInd'] = feat #将找到的最好划分特征和特征值作为划分节点，存入字典retTree
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val) #调用binSplitDataSet(),按找到的最佳划分特征feat和及其特征值val，将数据集dataSet划分为两个子树：lSet, rSet
    retTree['left'] = createTree(lSet, leafType, errType, ops) #左子树和右子树分别递归调用createTree()
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree  #返回构建的回归树

def isTree(obj):
    return (type(obj).__name__=='dict')

def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
    
def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree) #if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):#if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge: 
            print("merging")
            return treeMean
        else: return tree
    else: return tree
    
def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat