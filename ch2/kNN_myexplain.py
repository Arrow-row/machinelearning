'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label

@author: pbharrin
'''
import numpy as np  #numpy:科学计算包
import operator  #operator:运算符模块
from os import listdir

def classify0(inX, dataSet, labels, k):  #kNN算法实现
    '''
    dataSet:类别已知的样本特征值矩阵，NxM，对应的类别在向量labels中给出，N为样本特征维度，M为样本数量
    inX:输入向量，1xN，N为特征维度，是类别未知的需要预测的特征值向量
    labels:数据集的标签，1xM
    k:在样本数据集中选取k个与输入向量最相似的数据（k须为奇数）
    '''
    dataSetSize = dataSet.shape[0]  #dataSet为MxN，shape[0]获取其M值


    '''
    计算输入向量与样本集中各向量的欧式距离
    '''
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  #np.tile将inX扩展成包含M个inX的集合B(MxN)，再与样本集合dataSet做差，得到矩阵diffMat(MxN)
    sqDiffMat = diffMat**2  #diffMat中各元素平方得sqDiffMat（MxN）
    sqDistances = sqDiffMat.sum(axis=1)  #sqDiffMat矩阵按行求和(输入向量与样本集中各向量对应特征值的差值平方和)得sqDistances（Mx1）
    distances = sqDistances**0.5  #sqDistances中各元素开方得数组distances（Mx1）
    sortedDistIndicies = distances.argsort()  #将distances中元素排序（升序），argsort()返回的是排序后元素在原数组中的下标；sortedDistIndicies（Mx1）元素是distances元素下标

    '''
    统计前k个距离最近的样本的的label出现频率，存储到classCount中(key:label，value:count)，按value值排序后，返回value值最大的label
    '''
    classCount = {}      #声明字典变量classCount，用于记录前k个邻近样本中不同label出现的次数
    for i in range(k):   #取出前k个最相邻的样本数据标签，range(k)表示从0计数到k结束但不包括k
        voteIlabel = labels[sortedDistIndicies[i]]  #根据sortedDistIndicies[i]定位样本集中数据对应的标签，赋值给voteIlabel
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  #字典classCount中，若voteIlabel标签存在，则对应的数值+1，否则，将新标签加入字典并将对应数值+1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  #将classCount中数据按value值降序排列,返回重新排序后的列表。items()获取字典中的所有key-value对,返回可迭代对象dict_items;key表示取自可迭代对象、(进行一定运算后)用于进行比较的元素,这里通过itemgetter(1)取字典元素中的value值;reverse=True将排序规则设置为降序
    return sortedClassCount[0][0]  #取概率最大的标签

def createDataSet():    #createDataSet()创建数据集和标签 p19 
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])  #4组数据，每组包括2个特征值
    labels = ['A', 'A', 'B', 'B']
    return group, labels   #本函数返回变量group和labels

def file2matrix(filename):  #处理文本文件中的数据格式，输入为文件名字符串，输出为训练样本矩阵和类标签向量
    love_dictionary = {'largeDoses':3, 'smallDoses':2, 'didntLike':1}   #定义字典，数字3的含义为largeDoses，以此类推
    fr = open(filename)     #以只读方式打开filename文件，返回file对象fr
    arrayOLines = fr.readlines()  #读取完文件中所有内容，返回列表arrayOLines <class 'list'>，列表元素为文件的每行内容
    numberOfLines = len(arrayOLines)  #得到文件行数，即arrayOLines元素个数
    returnMat = np.zeros((numberOfLines, 3))  #创建用于返回的二维矩阵returnMat,shape=(numberOfLines, 3),行数numberOfLines为文件内容行数,列数3表示一条样本中特征的数量是3。数组元素的数据类型默认为numpy.float64
    classLabelVector = []     #用于返回的类型标签向量 <class 'list'>, 其中的元素取值为数字1,2,3,表示的含义见love_dictionary 
    index = 0
    for line in arrayOLines:  #按行依次处理文件内容，一次取一行
        line = line.strip()   #strip()删除每行首尾的空格后,生成的字符串赋给line <class 'str'>
        listFromLine = line.split('\t')   #用'\t'对line分片,生成的列表赋给listFromLine <class 'list'>,其中元素类型为字符串<class 'str'>
        returnMat[index, :] = listFromLine[0:3]  #returnMat矩阵中,第index行填充listFromLine列表中前3个元素。用到了矩阵索引和列表索引
        if(listFromLine[-1].isdigit()):   #isdigit()检测listFromLine中最后一个字符串元素是否只由数字0~9组成,是返回True,否返回False
            classLabelVector.append(int(listFromLine[-1]))  #Ture:将str转换为int后，追加到类型标签向量classLabelVector末尾。append()方法无返回值，但会修改原来的列表
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))  #False:以该str为key查找love_dictionary中所对应value值，追加到classLabelVector中
        index += 1
    return returnMat, classLabelVector  #本函数返回特征矩阵returnMat和类型标签向量classLabelVector


def autoNorm(dataSet):  #归一化特征值,将数字特征值转化为0到1的区间

    '''
    归一化公式:
    newValue=(oldValue-min)/(max-min)
    min和max分别是数据集中的最小特征值和最大特征值
    '''
    minVals = dataSet.min(0)  #将矩阵dataSet每列中最小元素放入minVals
    maxVals = dataSet.max(0)  #将矩阵dataSet每列中最大元素放入maxVals
    ranges = maxVals - minVals  #获取最大值与最小值特征值的差值，公式的分母部分
    normDataSet = np.zeros(np.shape(dataSet))  #创建存储归一化数据的矩阵normDataSet
    m = dataSet.shape[0]  #获取dataSet行数
    normDataSet = dataSet - np.tile(minVals, (m, 1))  #np.tile将minVals扩展成包含m个minVals的集合，再与样本集合dataSet中原数据做差，得到公式的分子部分的矩阵
    normDataSet = normDataSet/np.tile(ranges, (m, 1))  #np.tile将ranges扩展成包含m个ranges的集合,得到公式的分母部分的矩阵;代入公式得到归一化后的特征值矩阵;这里/是矩阵对应元素相除
    return normDataSet, ranges, minVals  #返回归一化后的特征值矩阵，差值向量，最小特征值向量

def datingClassTest():  #测试分类器准确度，采用错误率来评估
    hoRatio = 0.50      #hoRatio表示用于测试的数据占总数据的百分比
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')       #使用file2matrix函数，从datingTestSet2.txt文件中获取样本的特征值矩阵和标签向量
    normMat, ranges, minVals = autoNorm(datingDataMat)  #使用autoNorm函数，获取归一化特征值矩阵，差值向量，最小特征值向量
    m = normMat.shape[0]  #获取样本总数m
    numTestVecs = int(m*hoRatio)  #numTestVecs为用于测试准确度的样本数，即测试集样本数
    errorCount = 0.0  #初始化分类器出错计数器，后续每次预测出错，计数器+1
    for i in range(numTestVecs):  #依次取出测试集中第i个样本，range(numTestVecs)表示从0计数到numTestVecs结束但不包括numTestVecs
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)  #调用kNN分类器，inX:normMat中第i个行向量;dataSet:normMat中除去前numTestVecs条数据的数据集;labels:与dataSet数据对应的标签组成的向量;k:这里是样本特征数量,3;分类结果赋值给classifierResult,取值[1,3]
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))  #打印分类器预测值、原数据标签值
        if (classifierResult != datingLabels[i]): errorCount += 1.0  #若分类器预测出错，计数器errorCount+1
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))  #计算并打印分类器错误率
    print(errorCount)  #打印此次用于评估的测试集中，分类器预测出错次数

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input(\
                                  "percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream, ])
    classifierResult = classify0((inArr - \
                                  minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like this person: %s" % resultList[classifierResult - 1])

def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))
