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
    weights = np.ones((n, 1))  #特征的回归系数矩阵，系数初始化为1，weight为nx1矩阵（3x1）
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #样本特征矩阵与系数矩阵相乘后，传入sigmoid()计算结果分类结果为h，h是列向量，元素个数等于样本数（这里样本数为100）
        error = (labelMat - h)              #矩阵相减得到实际类型标签值与sigmoid()计算分类的误差，error是列向量，元素个数等于样本数
        weights = weights + alpha * dataMatrix.transpose()* error  #梯度上升算法迭代公式，按照差值的方向调整回归系数
    return weights

def plotBestFit(weights):  #本函数用于画出数据集散点图和Logistic回归最佳拟合分界线
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
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')  #以坐标向量(xcord1,ycord1)绘制类型1样本散点图，s为数值标量表示以相同的大小绘制所有标记，c='red'表示标记颜色红色，marker='s'表示标记样式为方形
    ax.scatter(xcord2, ycord2, s=30, c='green')  
    x = np.arange(-3.0, 3.0, 0.1)  #分界线的x轴坐标范围及划分步长
    #y = (-weights[0]-weights[1]*x)/weights[2]  #设置sigmoid函数为0,0是两个类别的分界处，0=W0X0+W1X1+W2X2，解出X1和X2的关系式，即分界线的方程，其中X0=1
    #y=np.array((-weights[0]-weights[1]*x)/weights[2])
    y=np.array((np.array((-weights[0]-weights[1]*x)/weights[2]))[0])
    ax.plot(x, y)  #绘制分界线
    plt.xlabel('X1'); plt.ylabel('X2')  #横轴X1，纵轴X2
    plt.show()  #显示图像

def stocGradAscent0(dataMatrix, classLabels):  #本函数实现随机梯度上升算法，返回回归系数。入参dataMatrix, classLabels可由loadDataSet()返回，二维dataMatrix是100×3的矩阵，每行表示一个样本，每列表示一个特征
    m, n = np.shape(dataMatrix)  #获取函数输入矩阵行m列n（m=100,n=3）
    alpha = 0.01  #alpha为向目标移动的步长(参见梯度上升算法迭代公式)
    weights = np.ones(n)   #所有回归系数初始化为1
    for i in range(m):  #依次获取每个样本
        h = sigmoid(sum(dataMatrix[i]*weights))  #取dataMatrix中样本i与系数向量相乘，得到经过sigmoid函数计算的样本i的类别，h是一个数值(gradAscent()中是向量)
        error = classLabels[i] - h  #样本i实际类别与计算类别误差error是一个数值(gradAscent()中是向量)
        weights = weights + alpha * error * dataMatrix[i]  #梯度上升算法迭代公式，按照差值的方向调整回归系数
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):  #改进的随机梯度上升算法，对stocGradAscent0()进行优化，numIter=150为增加的迭代次数参数，默认迭代150次
    m, n = np.shape(dataMatrix)  #获取函数输入矩阵行m列n（m=100,n=3）
    weights = np.ones(n)   #所有回归系数初始化为1 
    for j in range(numIter):  #j是迭代次数
        dataIndex = list(range(m))  #数据索引列表
        for i in range(m):  #i是样本点下标
            alpha = 4/(1.0+j+i)+0.0001    #alpha随着迭代次数增加而减小，但因为有常数项不会减至0，alpha每次减少1/(j+i)，非严格下降
            randIndex = int(np.random.uniform(0, len(dataIndex)))   #randIndex为[0,100)之间随机生成的整数，len(dataIndex)=100
            h = sigmoid(sum(dataMatrix[randIndex]*weights))  #通过randIndex随机选取dataMatrix中样本点来计算回归系数
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])  #从dataIndex删除已使用的样本
    return weights

def classifyVector(inX, weights):  #此为Logistics分类函数，inX为待分类数据特征向量，weights是梯度上升算法求得的最佳回归系数
    prob = sigmoid(sum(inX*weights))  #inX*weights输入sigmoid函数求得inX类别
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():  #训练分类器及检测错误率
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')  #打开测试集、训练集文件
    trainingSet = []; trainingLabels = []  #初始化训练集特征和标签列表
    for line in frTrain.readlines():  #依次循环frTrain中每一行。函数readlines()：读取完所有内容，返回每一行字符串组成的列表
        currLine = line.strip().split('\t')  #以'\t'划分当前行内容，去掉首尾空格后返回本行中特征值及标签值组成的列表，共22个元素
        lineArr = []
        for i in range(21):  #前21个元素为特征值，将其数值追加到特征值列表lineArr
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)  #特征值列表存入训练集矩阵trainingSet
        trainingLabels.append(float(currLine[21]))  #依次将样本标签值存入trainingLabels
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 1000)  #以训练集特征值及标签采用stocGradAscent1函数计算回归系数
    errorCount = 0; numTestVec = 0.0  #初始化分类测试错误次数和测试向量总数
    for line in frTest.readlines():  #依次循环frTrain中每一行，准备测试样本特征值
        numTestVec += 1.0  #统计测试集中样本数目
        currLine = line.strip().split('\t')  
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):  #调用分类函数classifyVector进行分类并判断分类结果是否正确
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)  #计算错误率
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():  #调用函数colicTest()10次并求结果的平均值
    numTests = 10; errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))
        