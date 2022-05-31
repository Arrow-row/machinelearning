import kMeans
from numpy import *

#10.1
#dataSet=kMeans.loadDataSet('testSet.txt')
datMat=mat(kMeans.loadDataSet('testSet.txt'))
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


#10.3