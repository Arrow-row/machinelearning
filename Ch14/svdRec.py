'''
Created on Mar 8, 2011

@author: Peter
'''
from numpy import *
from numpy import linalg as la  #linalg为numpy中的线性代数工具箱，其中包含的函数可以计算逆矩阵、求特征值、解线性方程组以及求解行列式等

def loadExData(): #测试数据,行号表示用户编号user，列号表示物品编号item, 均由0开始计数, 矩阵元素是用户给物品的评分(取值[1,5])，0表示用户对该物品未给出过评分(未使用过该物品)
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
    
def loadExData2(): #构建测试数据矩阵
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
    
def ecludSim(inA,inB): #相似度计算：使用欧式距离，并将取值归一化到(0,1)。inA和inB均为列向量
    return 1.0/(1.0 + la.norm(inA - inB)) #la.norm()函数计算向量的2范数

def pearsSim(inA,inB): #相似度计算：使用皮尔逊相关系数。皮尔逊相关系数取值范围为(-1,1),需要将取值归一化到(0,1)。inA和inB均为列向量
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA,inB): #相似度计算：使用余弦相似度。余弦相似度取值范围为(-1,1),需要将取值归一化到(0,1)。inA和inB均为列向量
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB) #la.norm()函数计算向量的2范数
    return 0.5+0.5*(num/denom)

def standEst(dataMat, user, simMeas, item): #给定相似度计算方法，计算用户对物品的估计评分值。数据矩阵dataMat(每行表示用户评分，每列表示不同物品),用户编号user,相似度计算方法simMeas,物品编号item
    n = shape(dataMat)[1] #矩阵的列数n为物品个数
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n): #遍历每个物品（矩阵每一列）
        userRating = dataMat[user,j] #获取用户user对物品j的评分
        if userRating == 0: continue #若评分为0，表示用户未做出评分，跳过当前物品，继续获取下一个物品
        overLap = nonzero(logical_and(dataMat[:,item].A>0, dataMat[:,j].A>0))[0] #获取item和j均有得分的元素行号overlap。 logical_and()对逐元素进行逻辑与运算;对于二维数组A，nonzero(A)得到一个长度为2的元组,它的第0个元素是数组中值不为0的元素的第0轴的下标，第1个元素则是第1轴的下标
        #print(type(overLap),overLap) #<class 'numpy.ndarray'> [0 3 4 5 6]
        if len(overLap) == 0: similarity = 0 #若两物品item,j的用户评分元素相与结果为0，则判断两物品不相似
        else: similarity = simMeas(dataMat[overLap,item], dataMat[overLap,j]) #否则调用相似度计算公式simMeas计算item,j两向量的相似度
        #print('the %d and %d similarity is: %f' % (item, j, similarity)) #打印两物品相似度
        simTotal += similarity #相似度累加
        ratSimTotal += similarity * userRating #相似度与用户评分乘积累加
    if simTotal == 0: return 0 #总相似度为0，退出循环
    else: return ratSimTotal/simTotal #返回针对user的关于物品item的评分预测值。通过除以总相似度simTotal，对相似度评分乘积ratSimTotal进行归一化，使得最后的评分值在0到5之间，可用于对预测值进行排序
    
def svdEst(dataMat, user, simMeas, item): #基于SVD的评分估计函数
    n = shape(dataMat)[1] #矩阵的列数n为物品个数
    simTotal = 0.0
    ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat) #调用la.svd()对矩阵dataMat做奇异值分解
    Sig4 = mat(eye(4)*Sigma[:4]) #建立对角矩阵。经过预先计算，前4个奇异值保存的信息超过90%
    #print(Sig4)
    xformedItems = dataMat.T * U[:,:4] * Sig4.I  #利用U矩阵将dataMat转换到低维空间    create transformed items
    for j in range(n): #在给定的用户对应行的所有元素上进行遍历
        userRating = dataMat[user,j] #获取用户user对物品j的评分
        if userRating == 0 or j==item: continue #若评分为0(表示用户未做出评分)，或遍历到的物品与待评估物品相同，跳过当前物品，继续获取下一个物品
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T) #调用simMeas，在低维空间下计算基于物品的相似度
        #print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity #相似度累加
        ratSimTotal += similarity * userRating #相似度与用户评分乘积累加
    if simTotal == 0: return 0 #总相似度为0，退出循环
    else: return ratSimTotal/simTotal #返回针对user的关于物品item的评分预测值

def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst): #本函数产生最高的N个推荐结果。参数：待处理数据矩阵dataMat，用户编号user,返回得分最高的前N个数据，相似度计算函数simMeas(默认cosSim)，评分预测值计算函数estMethod(默认standEst)
    unratedItems = nonzero(dataMat[user,:].A==0)[1] #unratedItems为用户未评分的物品在dataMat中的列下标 find unrated items 
    #print(type(unratedItems),unratedItems) #<class 'numpy.ndarray'> [1 2]
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = [] #初始化物品预测值列表
    for item in unratedItems: #遍历未进行评分的物品
        estimatedScore = estMethod(dataMat, user, simMeas, item) #调用estMethod引用的评分估计函数，计算用户user对item的评分预测值
        itemScores.append((item, estimatedScore)) #将物品编号及对应预测值以元组形式加入列表itemScores
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N] #用sorted()对itemScores进行排序，reverse=True表示降序，key=lambda jj: jj[1]表示排序的依据是itemScores中各元组的estimatedScore，即各物品的评分预测值。sorted()返回重新排序的列表，预测评分最大值在第一个元组。函数返回排序后的前N个元组

def printMat(inMat, thresh=0.8): #打印矩阵。矩阵包含浮点数，用thresh定义浅色深色
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print(1, end=''),
            else: print(0, end=''),
        print('')

def imgCompress(numSV=3, thresh=0.8): #实现图像压缩，并允许基于任意给定的奇异值数目来重构图像
    myl = []
    for line in open('0_5.txt').readlines(): #打开文本文件，遍历每一行
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow) #将文件每一行转换为列表(元素为int数值)，再存入列表myl
    myMat = mat(myl) #数据类型转换为mat
    print("****original matrix******")
    printMat(myMat, thresh) #调用printMat()打印原矩阵
    U,Sigma,VT = la.svd(myMat) #对myMat进行svd分解
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV): #由Sigma向量元素构建奇异值对角矩阵    construct diagonal matrix from vector
        SigRecon[k,k] = Sigma[k]
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:] #根据前3个奇异值和U、V矩阵，重构图像
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh) #打印重构的图像