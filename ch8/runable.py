from numpy import *
import matplotlib.pyplot as plt
import regression

'''
#8.1
xArr,yArr=regression.loadDataSet('ex0.txt')  # >>>xArr[0:2]   [[1.0, 0.067732], [1.0, 0.42781]]
#print(xArr[0:2],yArr[0:2]) #[[1.0, 0.067732], [1.0, 0.42781]] [3.176513, 3.816464]

ws=regression.standRegres(xArr,yArr) #由样本数据及原始标签计算回归系数w
xMat=mat(xArr)
yMat=mat(yArr)
yHat=xMat*ws #再由原始样本特征值矩阵xMat和算出的回归系数w计算预测标签值yHat

#绘制原始数据图像
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0]) #flatten()是numpy下的一个函数，返回多维数组/矩阵的一维副本;运算.A[0]将mat转换为array后，取第0轴;运算.T表示取转置
"""
>>> A=xMat[:,1][0:2].flatten().A
>>> A
array([[0.067732, 0.42781 ]])
>>> A=xMat[:,1][0:2].flatten().A[0]
>>> A
array([0.067732, 0.42781 ])
"""

#绘制计算出的最佳拟合直线
xCopy=xMat.copy()
xCopy.sort(0) #plot绘图，数据点需为有序
yHat=xCopy*ws
ax.plot(xCopy[:,1],yHat,'y',label='regress line')
plt.xlabel('x') 
plt.ylabel('y')
plt.legend()
plt.show()

#计算预测值和真实值的相关系数，用以评价拟合效果
#采用numpy提供的corrcoef()来计算相关性
yHat=xMat*ws
ralateRate=corrcoef(yHat.T,yMat)
print(ralateRate)
"""
[[1.         0.98647356]
 [0.98647356 1.        ]]
"""
'''

"""
#8.2
#不同k值下对xArr[0]的估计
xArr,yArr=regression.loadDataSet('ex0.txt') 
yp0=regression.lwlr(xArr[0],xArr,yArr,1.0)
print('k=1.0,yp0= ',yp0)
yp0=regression.lwlr(xArr[0],xArr,yArr,0.001)
print('k=0.001,yp0= ',yp0)

#计算数据集中所有点的估计值
#yHat=regression.lwlrTest(xArr,xArr,yArr,0.003)

#绘制数据集样本点的估计值和原始值
xMat=mat(xArr)
yMat=mat(yArr)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0]) #绘制原始数据图像。flatten()是numpy下的一个函数，返回多维数组/矩阵的一维副本;运算.A[0]将mat转换为array后，取第0轴;运算.T表示取转置

yHat,xCopy=regression.lwlrTestPlot(xArr,yArr,k=0.01)
ax.plot(xCopy[:,1],yHat,'red',label='lwlr line')
plt.xlabel('x') 
plt.ylabel('y')
plt.legend()
plt.show()
"""
"""
#8.4.1
abX,abY=regression.loadDataSet('abalone.txt')
#print(abX[4176])
#print(shape(abX),shape(abY)) 
ridgeWeights,lamMat=regression.ridgeTest(abX,abY) #调用ridgeTest()得到30个不同lambda对应的回归系数
#print(shape(ridgeWeights),shape(lamMat))

#绘制回归系数与log(lambda)关系图
fig=plt.figure()
ax=fig.add_subplot(111)
for i in range(shape(ridgeWeights)[1]):
	ax.plot(lamMat,ridgeWeights[:,i],label='w%d'%(i+1))
plt.xlabel('ln(lambda)')
plt.ylabel('ws')
plt.legend()
plt.show()
"""

#8.4.3
xArr,yArr=regression.loadDataSet('abalone.txt')
#regression.stageWise(xArr,yArr,0.01,200) #eps为步长，numIt为步数。eps=0.01,numIt=200
#regression.stageWise(xArr,yArr,0.001,5000) #eps=0.001,numIt=5000
#regression.stageWise(xArr,yArr,0.005,1000) #eps=0.005,numIt=1000
xMat=mat(xArr)
yMat=mat(yArr).T
xMat=regression.regularize(xMat)
yM=mean(yMat,0)
yMat=yMat-yM
weights=regression.standRegres(xMat,yMat.T)
print(weights.T)

