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

def createVocabList(dataSet): #本函数用于获取文档词汇表，dataSet是评论列表
    vocabSet = set([])  #创建一个空集vocabSet <class 'set'>
    for document in dataSet:  #依次获取dataSet中每条评论
        vocabSet = vocabSet | set(document) #vocabSet和set(document)两集合求并，最终dataSet中包含文档中出现的所有单词，元素不重复
    return list(vocabSet)  #返回所有不重复词组成的列表，即词汇表

def setOfWords2Vec(vocabList, inputSet):  #本函数将文档转换为词向量，vocabList为词汇列表，可由函数createVocabList()生成，inputSet为待转换文档，数据结构如postingList[0]，vocabList, inputSet类型均为list
    returnVec = [0]*len(vocabList)  #初始化输出的文档向量returnVec，其中所含元素都为0的向量，元素个数为len(vocabList)。 A=[0]*4，输出A：[0, 0, 0, 0]
    for word in inputSet:  #依次取出inputSet中单词
        if word in vocabList:  #若单词包含于词汇表vocabList，取出该词在词汇表中索引后，将词向量对应位置元素置1，表示词汇表中单词在文档中出现过
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)  #若词汇表中查不到当前单纯，打印表中不含该单词的信息
    return returnVec  #文档向量returnVec的每一元素为1或0，分别表示词汇表中的单词在输入文档中是否出现

def trainNB0(trainMatrix, trainCategory):  #本函数为朴素贝叶斯分类器训练函数，trainMatrix是转换为词向量后的文档矩阵，矩阵元素由returnVec组成,trainCategory是文档标签向量(classVec)
    numTrainDocs = len(trainMatrix)  #用于训练的文档个数
    numWords = len(trainMatrix[0])  #numWords为文档矩阵中每个文档样本词向量长度
    pAbusive = sum(trainCategory)/float(numTrainDocs)  #pAbusive为正向文档在所有文档中占比，二分类问题中，负向文档占比为1-pAbusive。sum(trainCategory)将trainCategory中元素求和。正向类型用1表示，0表示负向
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)      #change to np.ones()
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):  #遍历训练集中所有文档及其类别
        if trainCategory[i] == 1:  #当前文档属于类别1，获取用于求类型1文档中各特征条件概率的数据
            p1Num += trainMatrix[i]  #向量求和，计算类型1文档中各特征出现次数，作为分子
            p1Denom += sum(trainMatrix[i])  #所有类型1文档中包含的所有词汇之和作为分母
        else:  #当前文档属于类别0，获取用于求类型0文档中各特征条件概率的数据
            p0Num += trainMatrix[i]  #向量求和，计算类型0文档中各特征出现次数，作为分子
            p0Denom += sum(trainMatrix[i])  #所有类型0文档中包含的所有词汇之和作为分母
    p1Vect = np.log(p1Num/p1Denom)          #类型1文档各特征词数分别除以词汇总数，得到类型1文档中各特征词汇出现的概率向量p1Vect。log()计算自然对数 
    p0Vect = np.log(p0Num/p0Denom)          #类型0文档各特征词数分别除以词汇总数，得到类型0文档中各特征词汇出现的概率向量p0Vect    
    return p0Vect, p1Vect, pAbusive   #返回正向文档在所有文档中占比pAbusive、两类文档中各特征词汇概率向量p1Vect、p0Vect

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):  
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i, encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i, encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    trainingSet = range(50); testSet = []           #create test set
    for i in range(10):
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
            print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount)/len(testSet))
    #return vocabList, fullText

def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

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
