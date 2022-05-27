from numpy import *
import regTrees
import matplotlib.pyplot as plt

"""
testMat=mat(eye(4))
mat0,mat1=regTrees.binSplitDataSet(testMat,1,0.5)
print(mat0)
print(mat1)
"""


#dataSet='ex00.txt',fig.9-1
myMat0=mat(regTrees.loadDataSet('ex00.txt')) #ex2.txt中的训练数据，用于创建树
ws,xMat,yMat=regTrees.linearSolve(myMat0)
#print(shape(xMat),shape(yMat))
#print(xMat[0],xMat[1])

fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat[:,0].flatten().A[0],s=5)

retTree0=regTrees.createTree(myMat0)
print('regTree0:\n',retTree0)

#ax.scatter(xMat[:,1].flatten().A[0],yMat[:,0].flatten().A[0],s=5)

plt.xlabel('x') 
plt.ylabel('y')
#plt.legend()
plt.show()

"""
#dataSet='ex0.txt',fig.9-2
myDat1=regTrees.loadDataSet('ex0.txt')
myMat1=mat(myDat1)
retTree1=regTrees.createTree(myMat1)
print('\nregTree1:\n',retTree1)


#dataSet='ex2.txt',fig.9-3
myDat2=regTrees.loadDataSet('ex2.txt') #ex2.txt中的训练数据，用于创建树
myMat2=mat(myDat2)
myTree=regTrees.createTree(myMat2,ops=(0,1)) #创建最大的树用于剪枝
print('\nmyTree:\n',myTree)

myDatTest=regTrees.loadDataSet('ex2test.txt') #ex2test.txt为用于剪枝的测试数据
myMat2Test=mat(myDatTest)
retTree2=regTrees.prune(myTree,myMat2Test)
print('\nregTree2:\n',retTree2)
"""
"""
#dataSet='exp2.txt',fig.9-4
myMat3=mat(regTrees.loadDataSet('exp2.txt')) #ex2.txt中的训练数据，用于创建树
ws,xMat,yMat=regTrees.linearSolve(myMat3)
#print(shape(xMat),shape(yMat))
#print(xMat[0],xMat[1])

fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat[:,0].flatten().A[0],s=5)

retTree3=regTrees.createTree(myMat3,regTrees.modelLeaf,regTrees.modelErr, ops=(1,10)) #创建最大的树用于剪枝
print('\nregTree3:\n',retTree3)

x2Mat=xMat[nonzero(xMat[:,retTree3['spInd']+1]<=retTree3['spVal'])[0],:]
#print(x1Mat[0],x1Mat[1])
y2Hat=x2Mat*retTree3['right']
ax.plot(x2Mat[:,1],y2Hat,'r',label='line1')

x1Mat=xMat[nonzero(xMat[:,retTree3['spInd']+1]>retTree3['spVal'])[0],:]
#print(x1Mat[0],x1Mat[1])
y1Hat=x1Mat*retTree3['left']
ax.plot(x1Mat[:,1],y1Hat,'y',label='line2')

plt.xlabel('x') 
plt.ylabel('y')
plt.legend()
plt.show()
"""