'''
Created on Jan 8, 2011

@author: Peter
'''
from numpy import *

def loadDataSet(fileName):      #本函数从文件中解析由'\t'分隔的浮点型数据，fileName是待解析的文件名
    numFeat = len(open(fileName).readline().split('\t')) - 1 #获取单个样本的特征数量 
    dataMat = []; labelMat = [] #用于存放数据和标签的矩阵
    fr = open(fileName)  #调用open()，以只读形式打开ex0.txt文件，返回文件对象fr，可通过该对象调用文件相关函数对文件进行操作
    for line in fr.readlines(): #依次获取fr中每一行。fr.readlines()读取.txt文件中所有行并以每一行内容作为元素返回一个列表
        lineArr =[] #存放样本特征值的列表
        curLine = line.strip().split('\t') #以'\t'字符对每行字符串进行切片，strip()去除首尾空格，返回分割后的字符串组成的列表给curLine
        for i in range(numFeat): #遍历curLine中元素（不包含样本标签）
            lineArr.append(float(curLine[i])) #curLine中元素从字符串转换为浮点型后，依次追加到lineArr，构成样本特征值列表
        dataMat.append(lineArr) #由lineArr组成样本特征值矩阵
        labelMat.append(float(curLine[-1])) #将curLine中最后一个元素转换为浮点型数据后添加到labelMat,形成样本标签向量
    return dataMat,labelMat #返回特征值矩阵和样本标签

def standRegres(xArr,yArr): #本函数用于计算最佳拟合直线，xArr,yArr分别是loadDataSet()解析出的样本矩阵和标签向量
    xMat = mat(xArr) #X矩阵：list转换为matrix
    yMat = mat(yArr).T #Y向量：list转换为matrix并转置为列向量
    xTx = xMat.T*xMat #计算X^T*X
    if linalg.det(xTx) == 0.0: #计算并判断行列式X^T*X是否为0。linalg是numpy提供的线性代数库，函数det()用于计算行列式
        print("This matrix is singular, cannot do inverse") #若行列式为0，打印信息：这个矩阵是奇异的不能做逆。退出函数
        return
    ws = xTx.I * (xMat.T*yMat) #行列式不为0，用回归系数w的求解公式计算出w
    return ws #返回w

def lwlr(testPoint,xArr,yArr,k=1.0): #局部加权线性回归函数。testPoint是待预测样本点，xArr是训练样本特征值矩阵，yArr是训练样本真实标签，k是需要用户指定的参数，与权重值有关，默认设置为1.0
    xMat = mat(xArr); yMat = mat(yArr).T #输入数据转换为mat矩阵
    m = shape(xMat)[0] #m为样本数量
    weights = mat(eye((m))) #初始化权重矩阵weights为单位阵，阶数等于样本点个数，类型为mat
    for j in range(m):   #遍历数据集，计算每个样本点对应的权重                   #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]     #计算待预测点testPoint与当前样本的距离diffMat
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))  #使用高斯核，给待预测点附近的点赋予更高的权重:样本点与待预测点距离递增，权重以指数级衰减，输入参数k控制衰减速度
    #下面5行与OLS一样，使用lwlr公式计算回归系数ws
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws #返回待预测点的预测值

def lwlrTest(testArr,xArr,yArr,k=1.0):  #在待预测数据集testArr上应用lwlr()计算预测值
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def lwlrTestPlot(xArr,yArr,k=1.0):  #用lwlr()计算yHat,返回yHat和xCopy用以绘制拟合图像
    yHat = zeros(shape(yArr))       
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k)
    return yHat,xCopy

def rssError(yArr,yHatArr): #此函数计算预测值yHat的误差，使用平方误差    yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

def ridgeRegres(xMat,yMat,lam=0.2): #此函数实现岭回归算法
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam #xTx+lambda*I :单位矩阵I是型为nxn的方阵，n是样本特征数量，使得矩阵denom可逆
    if linalg.det(denom) == 0.0: #计算矩阵denom行列式并判断是否为0，若为0则直接退出
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat) #denom行列式非0，按岭回归中公式计算回归系数ws
    return ws #返回岭回归系数
    
def ridgeTest(xArr,yArr): #数据标准化处理后，适用岭回归计算回归系数
    xMat = mat(xArr); yMat=mat(yArr).T #数据转换为mat类型
    yMean = mean(yMat,0) #求样本标签值y的均值。mean(yMat,0)表示压缩行，对各列求均值
    yMat = yMat - yMean     #yMat中所有数据减去均值，原始yMat数据中心化，转换后的yMat均值为0
    #regularize X's
    xMeans = mean(xMat,0)   #对X矩阵按列求均值。mean(xMat,0)表示压缩行，对各列求均值     
    xVar = var(xMat,0)      #对X矩阵按列求方差。var()函数用于求方差         
    xMat = (xMat - xMeans)/xVar #原始xMat数据标准化。标准化过程：将所有数据减去平均值后再除以方差，调整得到的数据集均值为0，方差为1
    numTestPts = 30 #numTestPts控制lambda取值
    wMat = zeros((numTestPts,shape(xMat)[1])) #初始化30xn的系数矩阵
    for i in range(numTestPts): #exp(i-10)为30个不同的lambda取值，以指数级变化
        ws = ridgeRegres(xMat,yMat,exp(i-10)) #调用岭回归算法计算回归系数ws
        wMat[i,:]=ws.T #系数向量ws存入系数矩阵
    return wMat #返回系数矩阵

def regularize(xMat): #此函数用于对输入矩阵xMat按列正则化
    inMat = xMat.copy()  #获取输入的副本
    inMeans = mean(inMat,0)   #对xMat按列求均值。mean(inMat,0)表示压缩行，对各列求均值
    inVar = var(inMat,0)      #对xMat按列求方差。var()函数用于求方差
    inMat = (inMat - inMeans)/inVar #按列将数据减去平均值后再除以方差，调整得到的数据集均值为0，方差为1
    return inMat #返回正则化后的样本特征值矩阵

def stageWise(xArr,yArr,eps=0.01,numIt=100): #此函数前向逐步回归。xArr,yArr：输入特征值矩阵和标签向量；eps表示每次迭代需要调整的步长；numIt表示迭代次数
    xMat = mat(xArr); yMat=mat(yArr).T #数据转换为mat类型
    yMean = mean(yMat,0) #求样本标签值y的均值。mean(yMat,0)表示压缩行，对各列求均值
    yMat = yMat - yMean     #原始yMat数据中心化，转换后的yMat均值为0。也可以正则化      can also regularize ys but will get smaller coef
    xMat = regularize(xMat) #调用regularize()正则化输入矩阵
    m,n=shape(xMat) #获取输入矩阵的行、列值
    returnMat = zeros((numIt,n)) #testing code remove
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy() #为实现贪心算法建立ws的两份副本
    #贪心算法在所有特征上运行两次for循环，分别计算增加或减少该特征对误差的影响，初试误差设置为无穷，通过与所有误差比较后取最小
    for i in range(numIt): #
        print(ws.T) #打印当前系数向量ws
        lowestError = inf; 
        for j in range(n): #遍历样本点的每个特征
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

#def scrapePage(inFile,outFile,yr,numPce,origPrc):
#    from BeautifulSoup import BeautifulSoup
#    fr = open(inFile); fw=open(outFile,'a') #a is append mode writing
#    soup = BeautifulSoup(fr.read())
#    i=1
#    currentRow = soup.findAll('table', r="%d" % i)
#    while(len(currentRow)!=0):
#        title = currentRow[0].findAll('a')[1].text
#        lwrTitle = title.lower()
#        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
#            newFlag = 1.0
#        else:
#            newFlag = 0.0
#        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
#        if len(soldUnicde)==0:
#            print "item #%d did not sell" % i
#        else:
#            soldPrice = currentRow[0].findAll('td')[4]
#            priceStr = soldPrice.text
#            priceStr = priceStr.replace('$','') #strips out $
#            priceStr = priceStr.replace(',','') #strips out ,
#            if len(soldPrice)>1:
#                priceStr = priceStr.replace('Free shipping', '') #strips out Free Shipping
#            print "%s\t%d\t%s" % (priceStr,newFlag,title)
#            fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr,numPce,newFlag,origPrc,priceStr))
#        i += 1
#        currentRow = soup.findAll('table', r="%d" % i)
#    fw.close()
    
from time import sleep
import json
import urllib.request
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib.request.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if  sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: print('problem with item %d' % i)
    
def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)
    
def crossValidation(xArr,yArr,numVal=10):
    m = len(yArr)                           
    indexList = range(m)
    errorMat = zeros((numVal,30))#create error mat 30columns numVal rows
    for i in range(numVal):
        trainX=[]; trainY=[]
        testX = []; testY = []
        random.shuffle(indexList)
        for j in range(m):#create training set based on first 90% of values in indexList
            if j < m*0.9: 
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY)    #get 30 weight vectors from ridge
        for k in range(30):#loop over all of the ridge estimates
            matTestX = mat(testX); matTrainX=mat(trainX)
            meanTrain = mean(matTrainX,0)
            varTrain = var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain #regularize test with training params
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)#test ridge results and store
            errorMat[i,k]=rssError(yEst.T.A,array(testY))
            #print errorMat[i,k]
    meanErrors = mean(errorMat,0)#calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)]
    #can unregularize to get model
    #when we regularized we wrote Xreg = (x-meanX)/var(x)
    #we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = mat(xArr); yMat=mat(yArr).T
    meanX = mean(xMat,0); varX = var(xMat,0)
    unReg = bestWeights/varX
    print("the best model from Ridge Regression is:\n",unReg)
    print("with constant term: ",-1*sum(multiply(meanX,unReg)) + mean(yMat))