'''
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
'''
from numpy import *

def loadSimpData():  #构建简单数据集
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]]) #样本特征矩阵
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0] #类别标签
    return datMat,classLabels #返回样本矩阵和对应标签列表   <class 'numpy.matrixlib.defmatrix.matrix'>   <class 'list'>

def loadDataSet(fileName):      #本函数用于解析由'\t'分隔的文本文件中的样本特征   
    numFeat = len(open(fileName).readline().split('\t')) #open(fileName)打开文件fileName，readline()读取文件的一行，split('\t')将读取到的一行内容以'\t'分割，返回分割后的元素组成的列表，再以len获取列表长度，即单个样本中特征值个数
    dataMat = []; labelMat = [] #用于存储数据标签的矩阵
    fr = open(fileName) #以只读形式打开testSet.txt文件，返回文件对象fr，可通过该对象调用文件相关函数对文件进行操作
    for line in fr.readlines(): #依次获取fr中每一行。fr.readlines()读取文件中所有行并以行内容作为元素返回一个列表
        lineArr =[]
        curLine = line.strip().split('\t') #以'\t'字符对每行字符串进行切片，strip()去除首位空格，返回分割后的字符串列表给curLine
        for i in range(numFeat-1): #获取样本特征值列表lineArr
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr) #以lineArr组成样本特征值矩阵
        labelMat.append(float(curLine[-1])) #获取样本标签向量
    return dataMat,labelMat #返回样本特征值矩阵和标签向量

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq): #比较样本矩阵中第dimen维(列)元素(特征值)和阈值threshVal，对样本数据进行划分,dataMatrix是样本矩阵，dimen是特征维度，threshVal是阈值
    retArray = ones((shape(dataMatrix)[0],1)) #初始化array类型的返回数组retArray，元素全部设置为1
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0 #数组过滤
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    

def buildStump(dataArr,classLabels,D): #基于单层决策树构建弱分类器
    dataMatrix = mat(dataArr) #dataArr转换成matrix类型dataMatrix
    labelMat = mat(classLabels).T #classLabels转换成matrix类型后转置为labelMat
    m,n = shape(dataMatrix) #dataMatrix的型为mxn
    numSteps = 10.0 #此变量是什么用途
    bestStump = {} #存储给定权重D时所得到的最佳单层决策树信息
    bestClasEst = mat(zeros((m,1))) #记录最佳预测分类结果
    minError = inf #最小错误率初始化为正无穷（numpy模块中定义）    
    for i in range(n): #遍历样本所有特征维度  
        rangeMin = dataMatrix[:,i].min(); #第i维特征的最小值
        rangeMax = dataMatrix[:,i].max(); #第i维特征的最大值
        stepSize = (rangeMax-rangeMin)/numSteps #通过特征i取值中的最大最小值计算步长
        for j in range(-1,int(numSteps)+1): #j取值[-1,10]        loop over all range in current dimension
            for inequal in ['lt', 'gt']: #依次将stumpClassify()中的threshIneq设为'lt', 'gt'        
                threshVal = (rangeMin + float(j) * stepSize) #计算当前阈值
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#调用stumpClassify()，根据特征i计算各样本在阈值threshVal下的预测值，predictedVals中元素为标签1或-1     call stump classify with i, j, lessThan
                errArr = mat(ones((m,1))) #errArr用于标记预测出错的样本，初始化为元素均为1的mx1矩阵，与样本一一对应
                errArr[predictedVals == labelMat] = 0 #预测值与真实值相等的样本，errArr相应地置0
                weightedError = D.T*errArr  #根据样本权重向量计算带权误差，是带权重的样本误差之和    calc total error multiplied by D
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)) #打印当前维度，阈值，预测判别方式，带权误差
                if weightedError < minError: #带权误差小于设置的最小误差
                    minError = weightedError #最小误差更新为带权误差
                    bestClasEst = predictedVals.copy() #更新最佳预测分类结果为predictedVals
                    bestStump['dim'] = i #记录当前决策树信息
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst #返回最佳决策树，最小误差，最佳预测分类结果


def adaBoostTrainDS(dataArr,classLabels,numIt=40): #AdaBoost训练过程，采用单层决策树作为基分类器。dataArr是样本特征矩阵，classLabels是类别标签向量，numIt(默认值为40)是设置的最大迭代次数(若最后迭代numIt此后仍未达到理想准确度，以此作为退出循环的条件)
    weakClassArr = [] #列表weakClassArr用于存储基分类器信息
    m = shape(dataArr)[0] #m是样本数
    D = mat(ones((m,1))/m)   #初始化权重向量D，其中的元素代表当前每个样本在数据集中所占权重，初始值为1/m           init D to all equal
    aggClassEst = mat(zeros((m,1)))  #aggClassEst用于记录集成分类器给出的样本预测值
    for i in range(numIt): #迭代次数等于最终集成分类器包含的基分类器个数，也即weakClassArr中保存的决策树个数
        bestStump,error,classEst = buildStump(dataArr,classLabels,D) #调用buildStump()找到最佳单层决策树     build Stump
        print("D:",D.T) #打印当前权重向量D(转置后为行向量)
        alpha = float(0.5*log((1.0-error)/max(error,1e-16))) #应用基分类器权重计算公式，计算各个基分类器权重alpha，当error=0时，式中分母用1e-16代替避免发生除0溢出   calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha  #当前决策树字典中添加alpha
        weakClassArr.append(bestStump)  #当前决策树信息字典保存到weakClassArr列表              #store Stump Params in Array
        print("classEst: ",classEst.T)  #打印当前预测结果信息
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #     exponent for D calc, getting messy
        D = multiply(D,exp(expon))   #使用exp(expon)和各样本上一轮权重更新样本权重向量D，使得错误分类的样本的权重增加，正确分类的样本权重降低                 #Calc New D for next iteration
        D = D/D.sum() #对权重向量归一化处理
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst #根据alpha计算当前基分类器的预测值在集成分类器的预测值中的占比，并逐步累加，用以计算当前集成分类器的预测值，classEst、aggClassEst均是mx1向量
        print("aggClassEst: ",aggClassEst.T) #打印当前集成分类器预测值
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1))) #sign(aggClassEst)：通过符号函数将集成分类器预测值转换为类别标签，用-1，1表示,之后再与实际标签classLabels比较，若相等则对应位置元素置True，否则置为False，sign(aggClassEst) != mat(classLabels).T将输出元素值为True或False的mx1向量;再使用multiply将其与ones向量按元素相乘，True*1=1,False*1=0，最后输出aggErrors是元素为0、1的mx1向量
        errorRate = aggErrors.sum()/m #计算当前集成分类器错误率，aggErrors.sum()对aggErrors中元素求和
        print("total error: ",errorRate) #打印当前集成分类器错误率
        if errorRate == 0.0: break #若集成分类器错误率为0，提前退出for循环，集成分类器构建完毕；否则继续下一次迭代，增加基分类器个数，直到迭代次数达到预设的numIt再退出
    return weakClassArr #返回组成集成分类器的基分类器信息

def adaClassify(datToClass,classifierArr): #AdaBoost分类函数。datToClass是待分类样本，classifierArr是AdaBoost算法得到的基分类器信息列表
    dataMatrix = mat(datToClass) #测试集样本datToClass转换为mat矩阵    do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0] #获取待分类样本数量m
    aggClassEst = mat(zeros((m,1))) #aggClassEst用于记录集成分类器给出的样本预测值
    for i in range(len(classifierArr)): #遍历AdaBoost算法得到的基分类器
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq']) #调用stumpClassify()，使用训练出的基分类器i的参数，计算测试数据集的预测值  call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst #基分类器预测值乘以对应alpha，累加得到集成分类器预测值，classEst、aggClassEst均是mx1向量
        print(aggClassEst) #打印当前集成分类器预测值
    return sign(aggClassEst) #用符号函数将预测值转换为标签-1、1，返回数据集的预测结果向量，向量中一个元素表示一个样本的预测结果

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is: ",ySum*xStep)
