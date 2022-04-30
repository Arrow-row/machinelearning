'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
import numpy as np

def loadDataSet():  #获取样本数据的函数
    dataMat = []; labelMat = []  #数据矩阵和标签矩阵
    fr = open('testSet.txt')  #以只读形式打开testSet.txt文件，返回文件对象fr，可通过该对象调用文件相关函数对文件进行操作。testSet.txt中是100个样本的简单数据集，样本包含两个特征X1和X2和其标签
    for line in fr.readlines():  #依次获取fr中每一行。fr.readlines()读取文件中所有行并以行内容作为元素返回列表
        lineArr = line.strip().split()  #以空字符对每行字符串进行切片，并去除首位空格后，返回分割后的字符串列表给lineArr
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  #将lineArr前两个元素转换为float类型后，以列表形式追加到dataMat
        labelMat.append(int(lineArr[2]))  #lineArr中第3个元素转换为int型后，作为标签追加到labelMat
    return dataMat, labelMat  #返回数据矩阵和标签变量

def sigmoid(inX):  #sigmoid函数
    return 1.0 / (1 + np.exp(-inX))

def gradAscent(dataMatIn, classLabels):  #本函数实现梯度上升算法，返回回归系数。入参dataMatIn, classLabels可由loadDataSet()返回，二维dataMatIn是100×3的矩阵，每行表示一个样本，每列表示一个特征
    dataMatrix = np.mat(dataMatIn)             #入参dataMatIn转换为NumPy矩阵     
    labelMat = np.mat(classLabels).transpose() #入参classLabels转换为NumPy矩阵(行向量转置后变为列向量) labelMat为mx1矩阵
    m, n = np.shape(dataMatrix)  #获取函数输入矩阵行m列n（m=100,n=3）
    alpha = 0.001  #alpha为向目标移动的步长(参见梯度上升算法迭代公式)
    maxCycles = 500  #maxCycles迭代次数(参见梯度上升算法迭代公式)
    weights = np.ones((n, 1))  #特征的回归系数矩阵，系数初始化为1，weight为nx1矩阵
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #样本特征矩阵与系数矩阵相乘后，传入sigmoid()计算结果分类结果为h，h是列向量，元素个数等于样本数（这里样本数为100）
        error = (labelMat - h)              #矩阵相减得到实际类型标签值与sigmoid()计算分类的误差，error是列向量，元素个数等于样本数
        weights = weights + alpha * dataMatrix.transpose()* error  #梯度上升算法迭代公式，按照差值的方向调整回归系数
    return weights

def plotBestFit(weights):  #本函数用于画出数据集和Logistic回归最佳拟合直线
    import matplotlib.pyplot as plt  #Matplotlib是Python的绘图库,可绘制静态,动态,交互式的图表;Pyplot是Matplotlib的子库,提供了和MATLAB类似的绘图API
    dataMat, labelMat = loadDataSet()  #使用loadDataSet()返回的样本和标签数据
    dataArr = np.array(dataMat)  #dataMat转换为array数组
    n = np.shape(dataArr)[0]  #获取dataArr第0轴大小，这里是100
    xcord1 = []; ycord1 = []  #类型1样本的x、y轴坐标数据列表
    xcord2 = []; ycord2 = []  #类型2样本的x、y轴坐标数据列表
    for i in range(n):  #依次获取dataArr中样本
        if int(labelMat[i]) == 1:  #当前样本为类型1，将其坐标数据添加至xcord1、ycord1
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:  #当前样本为类型2，将其坐标数据添加至xcord1、ycord1
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()  #创建画布
    ax = fig.add_subplot(111)  #添加一个子图
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')  #以类型1样本(x,y)坐标向量绘制散点图，s为数值标量表示以相同的大小绘制所有标记，c='red'表示标记颜色红色，marker='s'表示标记样式为方形
    ax.scatter(xcord2, ycord2, s=30, c='green')  
    x = np.arange(-3.0, 3.0, 0.1)  #x轴坐标范围、刻度
    y = (-weights[0]-weights[1]*x)/weights[2]  #设置sigmoid函数为0,0是两个类别的分界处，0=W0X0+W1X1+W2X2，解出X1和X2的关系式，即分界线的方程，此处X0=1
    ax.plot(x, y)  #绘制分界线
    plt.xlabel('X1'); plt.ylabel('X2')  #绘制两类数据点
    plt.show()  #显示图像

def stocGradAscent0(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(np.random.uniform(0, len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))
        