import kMeans
from numpy import *
import matplotlib.pyplot as plt


#10.1
#dataSet=kMeans.loadDataSet('testSet.txt')
datMat=mat(kMeans.loadDataSet('testSet2.txt'))
'''
print(min(datMat[:,0]))
print(min(datMat[:,1]))
print(max(datMat[:,0]))
print(max(datMat[:,1]))

[[-5.379713]]
[[-4.232586]]
[[4.838138]]
[[5.1904]]
'''

randCent=kMeans.randCent(datMat,2)
'''
print(randCent)

[[ 3.6310122  -0.79331344]
 [-3.70009525  1.14486981]]
 '''

distEclud=kMeans.distEclud(datMat[0],datMat[1])
#print(distEclud) #5.184632816681332


myCentroids,clustAssing=kMeans.kMeans(datMat,4)
#print(myCentroids) #<class 'numpy.matrixlib.defmatrix.matrix'>

N=shape(myCentroids)[0] #N为质心/簇个数,myCentroids是mat类型
fig=plt.figure()
ax=fig.add_subplot(111)
#ax.scatter(datMat[:,0].flatten().A[0],datMat[:,1].flatten().A[0],s=10) #原始数据散点图

MARKER=["s","v","o","D"]
COLOR=["r","g","b","y"]
for i in range(N): #聚类结果散点图
    pltCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
    ax.scatter(pltCluster[:,0].flatten().A[0],pltCluster[:,1].flatten().A[0],marker=MARKER[i],c=COLOR[i],s=10)
ax.scatter(myCentroids[:,0].flatten().A[0],myCentroids[:,1].flatten().A[0],marker="x",c="k",s=60) #各簇质心

plt.xlabel('x') 
plt.ylabel('y')
#plt.legend()
plt.show()

"""
#10.3
datMat3=mat(kMeans.loadDataSet('testSet2.txt'))

fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(datMat3[:,0].flatten().A[0],datMat3[:,1].flatten().A[0],s=10) #原始数据散点图

centList,myNewAssments=kMeans.biKmeans(datMat3,3)
N=shape(centList)[0] #N为质心/簇个数,centList是mat类型
MARKER=["s","v","o"]
COLOR=["r","g","b"]

'''
for i in range(N): #聚类结果散点图
    pltCluster = datMat3[nonzero(myNewAssments[:,0].A==i)[0],:]
    ax.scatter(pltCluster[:,0].flatten().A[0],pltCluster[:,1].flatten().A[0],marker=MARKER[i],c=COLOR[i],s=10)
ax.scatter(centList[:,0].flatten().A[0],centList[:,1].flatten().A[0],marker="x",c="k",s=60) #各簇质心
'''

#print(myNewAssments[:,0])
#print(nonzero(myNewAssments[:,0].A==0)[0])
#print(Cluster1[:,0])

print(centList) #<class 'numpy.matrixlib.defmatrix.matrix'>   质心向量矩阵
'''
[[-0.45965615 -2.7782156 ]
 [ 2.93386365  3.12782785]
 [-2.94737575  3.3263781 ]]
 '''

plt.xlabel('x') 
plt.ylabel('y')
#plt.legend()
plt.show()
"""