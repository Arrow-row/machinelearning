'''
Created on Nov 4, 2010
Chapter 5 source file for Machine Learing in Action
@author: Peter
'''
from numpy import *
from time import sleep

def loadDataSet(fileName):  #从文件中解析样本数据的函数
    dataMat = []; labelMat = []  #样本特征矩阵和样本标签矩阵
    fr = open(fileName)  #以只读形式打开文件，返回文件对象fr，可通过该对象调用相关函数对文件进行操作
    for line in fr.readlines():  #依次获取文件中每一行。fr.readlines()读取文件中所有行并以行内容作为元素返回列表
        lineArr = line.strip().split('\t')  #以'\t'对每行字符串进行切片，并去除首位空格，返回分割后的字符串列表给lineArr
        dataMat.append([float(lineArr[0]), float(lineArr[1])])  #将lineArr前两个元素转换为float类型后，以列表形式追加到dataMat
        labelMat.append(float(lineArr[2]))  #lineArr中第3个元素转换为float型后，作为标签追加到labelMat
    return dataMat,labelMat  #返回样本特征矩阵和样本标签矩阵

def selectJrand(i,m):  #获取第2个alpha的下标j，i是外层循环已选的第1个alpha的下标，m是所有alpha的数目。简化版的SMO首先遍历每个alpha，然后在剩下的alpha集合中随机选择另一个alpha构建alpha对。
    j=i 
    while (j==i):  #要求i、j不相等
        j = int(random.uniform(0,m))  
    return j  #返回内层alpha下标j

def clipAlpha(aj,H,L): #对alpha进行剪辑
    if aj > H:  #若求出的aj大于区间最大值，则aj取区间最大值
        aj = H
    if L > aj:  #若求出的aj小于区间最小值，则aj取区间最小值
        aj = L
    return aj  #返回aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):  #简化版SMO算法。输入参数分别为：数据集，类别标签，常数C，容错率，退出前最大循环参数
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()  #输入dataMatIn转换为矩阵dataMatrix，classLabels转换为矩阵并转置得到列向量labelMat,类别标签向量中每行元素都和数据矩阵中的行一一对应
    b = 0
    m,n = shape(dataMatrix) #获取输入矩阵行m列n（m=100,n=2）
    alphas = mat(zeros((m,1))) #alpha列矩阵，m行1列，元素均初始化为0
    iter = 0 #记录alpha未成功更新的循环次数，只要有更新，该值就会被置0，连续多次未更新，该值会持续增加，当此变量值达到maxIter，也就是最大循环次数时，表明所有alpha更新完成，退出while循环，此函数结束运行
    while (iter < maxIter):
        alphaPairsChanged = 0 #记录alpha是否已经进行优化
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b  #fXi计算样本i的预测值,是一个数值,使用了预测函数g(x)=w*x+b,w=sum(ai*yi*xi),i=1,2...N;multiply(alphas,labelMat).T中,alphas,labelMat均是mx1的列向量,alphas是0向量,labelMat是标签向量,使用multiply使矩阵对应位置元素相乘,再使用.T将结果转置为1xm行向量;dataMatrix*dataMatrix[i,:].T中,dataMatrix与其第i行转置后做矩阵乘法,结果是mx1列向量;
            Ei = fXi - float(labelMat[i]) #预测结果与真实标签作差，得到预测误差Ei    if checks if an example violates KKT conditions
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)): #检测预测误差Ei是否超过预设的容忍度,若超过,则对样本i对应的alpha[i]进行优化;同时检查alpha值，使其不能等于0或C。alpha取值为(0,C)时，当前样本为支持向量，应满足labelMat[i]*Ei=0，这里toler是计算精度允许的误差范围
                j = selectJrand(i,m) #利用selectJrand()随机选择第2个alpha值，即alpha[j]
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b #fXj计算样本j的预测值
                Ej = fXj - float(labelMat[j]) #计算alpha[j]的预测误差
                alphaIold = alphas[i].copy() #保存alpha[i]的旧值
                alphaJold = alphas[j].copy() #保存alpha[j]的旧值
                if (labelMat[i] != labelMat[j]):  #根据SMO算法中子问题约束条件alpha取值[0,C],sum(alphas*labelMat)=0,以及标签值labelMat[i]、labelMat[j]关系,可以得到两种情况下alpha新值的取值范围[L,H],两种情况分别为labelMat[i]、labelMat[j]相等和不等
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print("L==H"); continue  #L、H相等不做任何改变，本次循环结束运行下一次for循环
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T  #计算系数2*K12-K11-K22
                if eta >= 0: print("eta>=0"); continue  
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta  #计算得出新的未剪辑的alphas[j]
                alphas[j] = clipAlpha(alphas[j],H,L)  #调用clipAlpha()对新的alphas[j]取值进行剪辑，限制其最小值在[L,H]之间
                if (abs(alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); continue  #若alphas[j]新旧值相比变化太小不够明显，则退出本次循环
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])  #计算新的alphas[i]值，alphas[i]和alphas[j]增量的大小相同，符号相反  
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T  #计算新的b1
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T  #计算新的b2
                if (0 < alphas[i]) and (C > alphas[i]): b = b1  #由于对alpha进行了剪辑(调用clipAlpha()),使得alpha取值为[0,C];取值范围在(0,C)的alpha对应的样例是支持向量,对应的bnew满足KKT条件
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0  #如果alpha=0或C,那么b1new和b2new均符合KKT条件,此时选择它们的中点作为bnew
                alphaPairsChanged += 1 #for循环内的语句执行到此，表示一对alpha成功被优化，标记alphaPairsChanged置为1，在此之前的任一continue都不会使本标记改变
                print("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        if (alphaPairsChanged == 0): iter += 1 #一次for循环后没有改变alpha对，迭代计数器iter+1;只要alpha有更新,iter就归0;连续迭代maxIter次后alpha对依然没有更新,退出while循环
        else: iter = 0
        print("iteration number: %d" % iter)
    return b,alphas  #返回alpha和b

def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K

class optStruct:  #结构化数据，便于使用
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  #用输入参数初始化类的属性值，self可看作java中的this。  Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #eCache为缓存矩阵，缓存误差值Ei，用于选择alpha时计算最大的Ei-Ej;eCache为m行2列，第1列是eCache是否有效的标识，第2列是实际的E值
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
        
def calcEk(oS, k): #本函数计算并返回预测误差Ek值，oS是optStruct类对象，k是迭代变量
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)  #计算预测值   为什么乘以oS.K[:,k]    注意K大写小写
    Ek = fXk - float(oS.labelMat[k])
    return Ek
        
def selectJ(i, oS, Ei):         #本函数用于选择内循环的alpha        this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #输入参数Ei值存入oS.eCache[i]并设置为有效的，有效的意思是已经计算好    set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]  #oS.eCache[:,0].A表示取matrix eCache中第0列，并将其从matrix转换为array，使用nonzero()[0]返回其中非0元素的索引值组成的列表
    if (len(validEcacheList)) > 1: #非0元素个数大于1，有效的Ei不止1个
        for k in validEcacheList:   #遍历非0元素的索引，用以找到使delta E最大的Ej   loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #Ei不再参与计算 
            Ek = calcEk(oS, k) #调用calcEk()计算误差Ek
            deltaE = abs(Ei - Ek)  #计算误差增量deltaE
            if (deltaE > maxDeltaE):  #若本次循环计算得到的最新deltaE大于上一轮的值，则更新maxK、maxDeltaE、Ej为当前计算所得
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej  #返回找到的内层循环的aloha下标maxK和对应误差值Ej
    else:   #刚开始计算时，误差缓存eCache中没有足够的有效值(第一列标志位均为0)，本算法采用的方式是随机选择一个alpha作内循环数据      in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m) #调用selectJrand()随机选取不同于i的下标作为内层alpha下标j
        Ej = calcEk(oS, j) #调用calcEk()计算误差Ej
    return j, Ej #返回找到的内层循环的aloha下标j和对应误差值Ej

def updateEk(oS, k): #重新计算Ek并存入缓存eCache         after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
        
def innerL(i, oS):   #本函数寻找合适的内循环下标j。i是外层循环变量，oS是optStruct类对象
    Ei = calcEk(oS, i)  #调用calcEk()计算预测误差Ei
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):  #选择误差oS.tol允许的支持向量
        j,Ej = selectJ(i, oS, Ei) #这里调用selectJ()选择内循环的alpha下标j，这里选择的alpha尽量满足 max(|Ei-Ej|)  (不同于简单版本SMO调用selectJrand())
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();  #保存alpha[i]、alpha[j]的旧值
        if (oS.labelMat[i] != oS.labelMat[j]):   #根据SMO算法中子问题约束条件alpha取值[0,C],sum(alphas*labelMat)=0,以及标签值labelMat[i]、labelMat[j]关系,可以得到两种情况下alpha新值的取值范围[L,H],两种情况分别为labelMat[i]、labelMat[j]相等和不等
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print("L==H"); return 0  #若L=H，当前j不是要找的内层alpha下标值，结束本次循环
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #使用核函数后的eta计算,eta=2*K12-K11-K22,用于后续求alpha时做分母
        if eta >= 0: print("eta>=0"); return 0  #若eta>=0，当前j不是要找的内层alpha下标值，结束本次循环
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta #计算得出新的未剪辑的alphas[j]，并以此更新对象oS中的alphas[j]
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L) #调用clipAlpha()对新的alphas[j]取值进行剪辑，限制其取值在[L,H]之间，并以此更新对象oS中的alphas[j]
        updateEk(oS, j) #oS中的alphas[j]更新后，重新计算对应样本的预测值和误差值，并调用updateEk()更新oS.eCache[j]，标记该alpha已经过更新，且记录下新的Ej        added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0  #若alphas[j]新旧值相比变化太小，当前j不是要找的内层alpha下标值，结束本次循环
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j]) #根据新的alphas[j]计算新的alphas[i]值，并以此更新对象oS中的alphas[i]（alphas[i]和alphas[j]增量的大小相同，符号相反）
        updateEk(oS, i) #oS中的alphas[i]更新后，重新计算对应样本的预测值和误差值，并调用updateEk()更新oS.eCache[i]，标记该alpha已经过更新，且记录下新的Ei                
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]  #计算新的b1
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]  #计算新的b2
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1  #任一取值范围在(0,C)的alpha对应的样例都是支持向量，对应约束条件得出的b均为超平面的正确参数
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0  #如果alpha=0或C,那么b1new和b2new均符合KKT条件,此时选择它们的中点作为bnew
        return 1  #循环结束，本次循环已找到满足要求的j
    else: return 0  #alpha[i]不合理，退出循环，需要重新选择另一外循环alpha

def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #SMO算法实现
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)  #初始化optStruct对象oS
    iter = 0 #记录alpha未成功更新的循环次数，只要有更新，该值就会被置0，连续多次未更新，该值会持续增加，当此变量值达到maxIter，也就是设置的最大循环次数时，表明所有alpha无需再更新，退出while循环，此函数结束运行
    entireSet = True #entireSet决定是否要完整遍历alpha列表
    alphaPairsChanged = 0 #记录当前alpha对是否已成功优化 
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):  #while循环退出条件：1.迭代次数iter超过指定的最大值maxIter  2.遍历整个集合都未对任意alpha做出修改
        alphaPairsChanged = 0
        if entireSet:   #首次迭代，entireSet=True，外循环的alpha[i]均未更新，需要遍历整个列表    ##### 为什么这里需要完全遍历    go over all
            for i in range(oS.m):  
                alphaPairsChanged += innerL(i,oS)  #调用innerL(),若成功更新了oS.alphas[i]、oS.alphas[j]以及其他参数Ei、Ej、b，alphaPairsChanged+1
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)) #打印本次循环结果
            iter += 1 #不管是否有alpha对成功被优化，迭代计数器iter均会+1
        else: #entireSet=False         go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]  #减小需要优化计算的alpha范围，遍历所有非边界alpha值，也就是不在边界0或C上的值     
            for i in nonBoundIs: #
                alphaPairsChanged += innerL(i,oS)  #调用innerL(),若成功更新了oS.alphas[i]、oS.alphas[j]以及其他参数Ei、Ej、b，alphaPairsChanged+1
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)) #打印本次循环结果
            iter += 1 #不管是否有alpha对成功被优化，迭代计数器iter均会+1
        if entireSet: entireSet = False #entireSet=True，表明当前是第一次迭代，之后entireSet置为false,下一次迭代不需遍历整个alpha列表，   toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  #entireSet=False,但是第iter次迭代没有成功更新的alpha对，下一次需要遍历整个列表  
        print("iteration number: %d" % iter)  #打印当前已进行的总的迭代次数
    return oS.b,oS.alphas #返回alpha和b

def calcWs(alphas,dataArr,classLabels): #基于alpha计算w,应用w的求解公式 w=sum(alpha[i]*y[i]*x[i]),i=1,2..N
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)  #m为样本数量，n为特征数量对于测试文本testSet.txt中的数据，计算得到 m = 100 , n = 2
    w = zeros((n,1)) #array类型数组,shape=(n, 1) 
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)  #求得w  <class 'numpy.ndarray'>
    return w

def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print("the test error rate is: %f" % (float(errorCount)/m))    
    
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels    

def testDigits(kTup=('rbf', 10)):
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] 
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print("the test error rate is: %f" % (float(errorCount)/m)) 


'''#######********************************
Non-Kernel VErsions below
'''#######********************************

class optStructK:
    def __init__(self,dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        
def calcEkK(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek
        
def selectJK(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEkK(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEkK(oS, j)
    return j, Ej

def updateEkK(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEkK(oS, k)
    oS.eCache[k] = [1,Ek]
        
def innerLK(i, oS):
    Ei = calcEkK(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJK(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print("L==H"); return 0
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEkK(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEkK(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoPK(dataMatIn, classLabels, C, toler, maxIter):    #full Platt SMO
    oS = optStructK(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):        
                alphaPairsChanged += innerLK(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerLK(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas