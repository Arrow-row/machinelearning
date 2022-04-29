'''
Created on Oct 19, 2010

@author: Peter
'''
import numpy as np

def loadDataSet():  #创建简单数据样本的函数
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],   #postingList是进行词条切分后的文档列表，包含6条评论数据，每条是组成该评论的单词组成的列表
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]    #postingList中6条评论的标签，1表示负面评论，0表示正面评论，人为做的分类标签
    return postingList, classVec   #返回postingList, classVec

def createVocabList(dataSet): #本函数用于获取文档词汇表，dataSet是评论列表，类型为list
    vocabSet = set([])  #创建一个空集vocabSet <class 'set'>
    for document in dataSet:  #依次获取dataSet中每条评论
        vocabSet = vocabSet | set(document) #vocabSet和set(document)两集合求并，最终vocabSet中包含文档中出现的所有单词，元素不重复
    return list(vocabSet)  #返回所有不重复词组成的列表，即词汇表

def setOfWords2Vec(vocabList, inputSet):  #本函数将文档转换为词向量，vocabList为词汇列表，可由函数createVocabList()生成，inputSet为待转换文档，数据结构如postingList[0]，vocabList, inputSet类型均为list
    returnVec = [0]*len(vocabList)  #初始化用于返回的文档向量returnVec，其中所含元素都为0，元素个数为len(vocabList)。  eg: A=[0]*4，输出A=[0, 0, 0, 0]
    for word in inputSet:  #依次取出inputSet中单词
        if word in vocabList:  #若当前单词包含于词汇表vocabList，取出该词在词汇表中索引后，将词向量对应位置元素置1，表示词汇表中单词在文档中出现过，同一单词多次出现只计1次
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)  #若词汇表中查不到当前单词，打印表中不含该单词的信息
    return returnVec  #文档向量returnVec的每一元素为1或0，分别表示词汇表中的单词在输入文档中是否出现

def bagOfWords2VecMN(vocabList, inputSet):  #本函数是对setOfWords2Vec()的修改，以实现朴素贝叶斯词袋模型。vocabList为词汇列表，inputSet为待转换文档单词组成的列表
    returnVec = [0]*len(vocabList)  #初始化用于返回的文档向量returnVec，其中所含元素都为0，元素个数为len(vocabList)
    for word in inputSet:  #依次取出inputSet中单词
        if word in vocabList:  #若当前单词包含于词汇表vocabList，词向量对应位置元素+1，同一单词多次出现依次累加
            returnVec[vocabList.index(word)] += 1
    return returnVec  #返回文档向量returnVec，其中元素表示词汇表中的单词在输入文档中出现次数

def trainNB0(trainMatrix, trainCategory):  #本函数为朴素贝叶斯分类器训练函数，trainMatrix <class 'numpy.ndarray'>是转换为词向量后的文档矩阵，矩阵元素由returnVec组成;trainCategory <class 'numpy.ndarray'>是文档标签向量(classVec)
    numTrainDocs = len(trainMatrix)  #用于训练的文档个数
    numWords = len(trainMatrix[0])  #numWords为文档矩阵中每个文档样本词向量长度
    pAbusive = sum(trainCategory)/float(numTrainDocs)  #pAbusive为正向文档在所有文档中占比，二分类问题中，负向文档占比为1-pAbusive。sum(trainCategory)将trainCategory中元素求和。正向类型用1表示，0表示负向
    #p0Num=np.zeros(numwords);p1Num=np.zeros(numwords)  #两种类型文档各特征出现次数的统计向量，做条件概率计算的分子，初始化为0
    #p0Denom=0.0;p1Denom=0.0   #两种类型文档特征总数，做条件概率计算的分母，初始化为0
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)  #概率相乘时，避免其中一个概率值为0导致最后的乘积也为0对分类结果造成影响，可以将所有词的出现数初始化为1，并将分母初始化为2。
    p0Denom = 2.0; p1Denom = 2.0                           
    for i in range(numTrainDocs):  #遍历训练集中所有文档及其类别
        if trainCategory[i] == 1:  #当前文档属于类别1，获取用于求类型1文档中各特征条件概率的数据
            p1Num += trainMatrix[i]  #向量求和，计算类型1文档中各特征出现次数，作为分子
            p1Denom += sum(trainMatrix[i])  #所有类型1文档中包含的所有词汇之和作为分母
        else:  #当前文档属于类别0，获取用于求类型0文档中各特征条件概率的数据
            p0Num += trainMatrix[i]  #向量求和，计算类型0文档中各特征出现次数，作为分子
            p0Denom += sum(trainMatrix[i])  #所有类型0文档中包含的所有词汇之和作为分母
    #p1Vect=p1Num/p1Denom  #类型1文档各特征词出现次数分别除以词汇总数，得到类型1文档中各特征词汇出现的概率向量p1Vect
    #p0Vect=p0Num/p0Denom  #类型0文档各特征词出现次数分别除以词汇总数，得到类型0文档中各特征词汇出现的概率向量p0Vect
    p1Vect = np.log(p1Num/p1Denom)    #概率相乘时，大部分因子都非常小导致程序会下溢出或者得到不正确的答案，一种解决办法是对乘积取自然对数。log()计算自然对数 
    p0Vect = np.log(p0Num/p0Denom)              
    return p0Vect, p1Vect, pAbusive   #返回两类文档中各特征词汇概率向量p1Vect、p0Vect，正向文档在所有文档中出现概率pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):  #vec2Classify是待分类向量，p0Vec, p1Vec, pClass1是函数trainNB0()返回的三个概率
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    #计算vec2Classify属于类型1的概率p1。由于使用了对数，概率相乘运算变为相加运算
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)  ##计算vec2Classify属于类型0的概率p0
    if p1 > p0:  #vec2Classify类型取p1/p0中概率较大者
        return 1
    else:
        return 0

def testingNB():  #贝叶斯分类器测试函数
    listOPosts, listClasses = loadDataSet()  #加载数据，listOPosts是进行词条切分后的评论集，listClasses是对应的类别
    myVocabList = createVocabList(listOPosts)  #myVocabList是listOPosts的词汇表
    trainMat = []  #存储词向量
    for postinDoc in listOPosts:  #将listOPosts中每条文档转换为词向量后，追加到矩阵trainMat
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))  
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))  #调用trainNB0(),入参是numpy.ndarray类型，使用数据listOPosts, listClasses训练分类器
    testEntry = ['love', 'my', 'dalmation']  #testEntry为待测试文档
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))  #调用setOfWords2Vec()将待测试文档转化为词向量
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))  #调用classifyNB()对文档进行分类后，打印分类结果
    testEntry = ['stupid', 'garbage']  #第二个测试文档
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

def textParse(bigString):    #本函数进行文本解析。将邮件文档解析后，返回其中所有长度大于2的单词组成的列表，单词可重复，不包括标点，字母均小写   input is big string, #output is word list
    import re  #正则表达式模块re，提供 Perl 风格的正则表达式模式
    listOfTokens = re.split(r'\W+', bigString)  #字符串前面的r表示原始字符串，不需要识别转义；split(r'\W+', bigString)以单词、数字之外的字符串对文本bigString进行切分
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]  #去掉空格，大写字母转为小写后，返回文档划分后词汇列表

def spamTest():
    docList = []; classList = []; fullText = []  #email/下有50个文件，列表docList中包含50个list，每个list由文件中单词组成，形如postingList；classList是50个文档的类型向量，元素为0或1；列表fullText中元素由50个文件所有单词组成
    for i in range(1, 26):  #email/spam/和email/ham/下各有25个.txt文件，依次读取第i个
        wordList = textParse(open('email/spam/%d.txt' % i, encoding="ISO-8859-1").read())  #读取文件所有内容后，经过textParse()解析为单词列表wordList
        docList.append(wordList)  #wordList以单个列表形式追加到docList
        fullText.extend(wordList)  #wordList以全部单词形式追加到fullText
        classList.append(1)  #classList依次记录文档类别1或0
        wordList = textParse(open('email/ham/%d.txt' % i, encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  #调用createVocabList()，用docList创建词汇表
    trainingSet = range(50); testSet = []           #create test set
    for i in range(10):  #向testSet中随机添加10个元素作为测试集索引，训练集trainingSet中索引不包括测试集中索引，两数据集元素均取值[0,50)
        randIndex = int(np.random.uniform(0, len(trainingSet)))  #randIndex为[0,50)之间随机生成的整数，len(trainingSet)=50
        testSet.append(trainingSet[randIndex])
        del(list(trainingSet)[randIndex])
    trainMat = []; trainClasses = []  #trainMat为训练集文档，trainClasses为文档对应的类型
    for docIndex in trainingSet:  #根据trainingSet中索引，获取docList中对应文档作为训练集  train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))  #将文档转换为词袋向量后，追加到trainMat
        trainClasses.append(classList[docIndex])  #trainClasses为当前文档对应类别
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))  #调用trainNB0()，用trainMat中40个.txt文件及其类别标签训练分类器
    errorCount = 0  #统计测试集文档分类错误次数
    for docIndex in testSet:  #根据testSet中索引，获取docList中对应文档作为测试集，用贝叶斯分类器classifyNB()对测试集文档进行分类
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])  #依次取出测试文档转换为词袋向量
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:  #贝叶斯分类器分类结果与实际不同，错误次数errorCount+1
            errorCount += 1
            print("classification error", docList[docIndex])  #打印分类错误的文档
    print('the error rate is: ', float(errorCount)/len(testSet))  #打印错误率
    #return vocabList, fullText

def calcMostFreq(vocabList, fullText):  #本函数遍历词汇表中的每个词并统计其在文本中出现的次数
    import operator
    freqDict = {}  #单词表词汇出现频度字典
    for token in vocabList:  
        freqDict[token] = fullText.count(token)  #统计每个词在fullText中出现次数，键值对存于freqDict
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)  #freqDict中数据按value值降序排列，返回重新排序后的列表，列表中元素是key-value的元组(按value降序)
    return sortedFreq[:30]  #函数返回sortedFreq中前30个元素

def localWords(feed1, feed0):
    import feedparser
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList, fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet = []           #create test set
    for i in range(20):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(list(trainingSet)[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount)/len(testSet))
    return vocabList, p0V, p1V

def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []; topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0: topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0: topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])
