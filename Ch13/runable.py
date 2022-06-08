import pca
import matplotlib.pyplot as plt
from numpy import *

"""
#13.2.2
dataMat=pca.loadDataSet('testSet.txt')
#print('dataMat: ',dataMat)
'''
dataMat:  [[10.235186 11.321997]
 [10.122339 11.810993]
 [ 9.190236  8.904943]]
'''
lowDMat,reconMat=pca.pca(dataMat,1)
#lowDMat,reconMat,meanRemoved=pca.pca(dataMat,1)
#print(shape(lowDMat)) #(1000, 1)
yaxis=mat(zeros((1000,1)))

fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=10,label='original data') #原始数据散点图
#ax.scatter(meanRemoved[:,0].flatten().A[0],meanRemoved[:,1].flatten().A[0],marker='^',s=10,c='y',label='meanRemoved data') #数据中心化散点图
#ax.scatter(lowDMat[:,0].flatten().A[0],yaxis[:,0].flatten().A[0],marker='o',s=10,c='g',label='low dim mat') #低维数据散点图
ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=10,c='red',label='the 1st principal compo') #主成分散点图

plt.xlabel('fet1') 
plt.ylabel('fet2')
plt.legend()
plt.show()
"""

#13.3
dataMat = pca.replaceNanWithMean() #调用replaceNanWithMean() 将数据集中所有的NaN替换成平均值

meanVals = mean(dataMat, axis=0)
meanRemoved = dataMat - meanVals #数据集中心化
covMat = cov(meanRemoved, rowvar=0) #中心化后计算协方差
eigVals,eigVects = linalg.eig(mat(covMat)) #协方差矩阵的特征值和特征向量
eigValInd = argsort(eigVals) #特征值由小到大排序，argsort()返回排序后元素在原列表中的索引      sort, sort goes smallest to largest
eigValInd = eigValInd[::-1] #特征值元素索引由升序改为降序
sortedEigVals = eigVals[eigValInd] #由索引获取原列表中的特征值
#print(type(sortedEigVals),shape(sortedEigVals)) #<class 'numpy.ndarray'> (590,)
total = sum(sortedEigVals) #特征值列表中的所有元素求和
varPercentage = sortedEigVals/total*100 #计算每个特征值在total中所占百分比
print(varPercentage[:20])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(1, 21), varPercentage[:20], marker='^') #绘制前20个主成分占总方差的百分比
plt.xlabel('Principal Component Number')
plt.ylabel('Percentage of Variance')
plt.show()
