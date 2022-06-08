import pca
import matplotlib.pyplot as plt
from numpy import *

#13.2.2
dataMat=pca.loadDataSet('testSet.txt')
#print('dataMat: ',dataMat)
'''
dataMat:  [[10.235186 11.321997]
 [10.122339 11.810993]
 [ 9.190236  8.904943]]
'''
lowDMat,reconMat=pca.pca(dataMat,1)
print(shape(lowDMat))

fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=10,label='original data') #原始数据散点图
ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=10,c='red',label='1st principal compo') #主成分散点图
plt.xlabel('fet1') 
plt.ylabel('fet2')
plt.legend()
plt.show()

