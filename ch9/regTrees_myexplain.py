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

def regErr(dataSet): #计算数据集dataSet的总方差，用于估计数据的一致性
    return var(dataSet[:,-1]) * shape(dataSet)[0] #返回值为总方差，采用均方差乘以数据集中样本的个数得到总方差

def linearSolve(dataSet):   # helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#初始化特征矩阵X和标签向量Y
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#X的第1列全为1，后n-1列存储dataSet中样本特征值；Y矩阵存储dataSet最后一列的标签值
    xTx = X.T*X 
    if linalg.det(xTx) == 0.0: #由最小二乘法计算回归系数ws，要求xTx可逆
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y #返回回归系数、特征矩阵X和标签向量Y

def modelLeaf(dataSet): #生成模型树叶节点的线性模型。在dataSet数据集上调用linearSolve()并返回回归系数ws     create linear model and return coeficients 
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet): #使用线性回归计算预测值yHat，返回预测值与真实值之间的的误差，在chooseBestSplit()中调用后用于找到数据集的最佳划分
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)): #此函数在给定的误差计算方法下，找到数据集的最佳二元划分方式，返回达到最佳划分的特征和特征值。输入参数：数据集dataSet，leafType是对叶节点建立函数的引用，errType是对总方差计算函数的引用，包含树构建所需其他参数的元组ops
    tolS = ops[0]; tolN = ops[1]  #用户指定的参数，用于控制函数停止。tolS是容许的误差下降值，tolN是切分的最少样本数
    #if all the target variables are the same value: quit and return value
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #退出条件1:子集中只有一类标签,无需划分数据集   
        return None, leafType(dataSet) #leafType是对创建叶节点的函数regLeaf()的引用
    m,n = shape(dataSet) #获取数据集行列数
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet) #调用regErr()计算当前数据集的总方差
    bestS = inf; bestIndex = 0; bestValue = 0  #初始化最佳划分的总方差、特征索引、特征值
    for featIndex in range(n-1): #遍历所有特征
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]): #遍历当前特征的所有特征值，特征值去重
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal) #调用binSplitDataSet(),用第n个特征的第i个特征值二分数据集dataSet为mat0、mat1
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue #子集样本点数 < tolN,退出当前内层for循环，执行下一次内层for循环
            newS = errType(mat0) + errType(mat1) #两个子集的总方差之和存于newS
            if newS < bestS: #若当前方差小于最小误差，则将当前切分设定为最佳切分并更新最小误差
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS:  #比较划分后的数据集与原数据集方差大小，检验划分后的数据一致性提升是否足够明显
        return None, leafType(dataSet) #退出条件2：划分后数据一致性提升不够明显,无需划分数据集
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue) #数据一致性提升足够明显，则使用最佳划分将dataSet分为mat0, mat1
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #退出条件3：#最佳划分后得到的子集样本点数 < tolN,无需划分数据集
        return None, leafType(dataSet)
    return bestIndex,bestValue #返回最佳划分所用的特征和特征值

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

def isTree(obj): #判断当前处理的节点是否为叶节点，若是叶节点，返回False
    return (type(obj).__name__=='dict') #输入对象类型名是否为'dict'，是返回True，否则返回False

def getMean(tree): #递归查找左右字数的叶子节点，如果找到，计算两叶子节点的均值并返回
    if isTree(tree['right']): tree['right'] = getMean(tree['right']) 
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0 #返回找到的两叶子节点的均值
    
def prune(tree, testData): #剪枝算法实现。tree是待剪枝的树, testData是剪枝所需的测试数据
    if shape(testData)[0] == 0: return getMean(tree) #确认测试集非空    if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):#当前节点不是叶节点，则调用binSplitDataSet()将测试集testData二分为两个子集lSet, rSet    if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet) #左子树递归调用prune()进行剪枝
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet) #右子树递归调用prune()进行剪枝
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']): #左右节点均是叶节点
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal']) #调用binSplitDataSet()将测试集testData二分
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2)) #剪枝前误差：左右叶节点中，预测值与测试集真实值之差的平方，相加后，得到当前子树的剪枝前误差
        treeMean = (tree['left']+tree['right'])/2.0 #左右叶节点代表的预测值求均值
        errorMerge = sum(power(testData[:,-1] - treeMean,2)) #剪枝后误差：子树预测均值与测试集真实值之差的平方
        if errorMerge < errorNoMerge: #剪枝后误差小于剪枝前，对叶节点进行合并，原子树预测均值作为新的叶节点的预测值
            print("merging")
            return treeMean
        else: return tree #剪枝后误差未减小，则无需剪枝，不合并直接返回原来的子树
    else: return tree #左右节点不都是叶节点，不进行合并
    
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