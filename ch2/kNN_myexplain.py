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

def classify0(inX, dataSet, labels, k):  #kNN算法
    '''
    dataSet:类别已知的样本数据集(不带标签)，NxM，N为样本特征维度，M为数据集中样本数量
    inX:与样本数据集作比较的输入向量，1xN，N为特征维度
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


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))   #element wise divide
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)

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
