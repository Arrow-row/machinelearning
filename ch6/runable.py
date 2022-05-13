import svmMLiA
from numpy import *


dataArr,labelArr = svmMLiA.loadDataSet('testSet.txt')  #测试文本共100条数据，类别标签为+1和-1两类
#b,alphas = svmMLiA.smoSimple(dataArr,labelArr,0.6,0.001,40)
b,alphas = svmMLiA.smoP(dataArr,labelArr,0.6,0.001,40)


print("b*=")
print(b)
print("alpha矩阵中大于0的元素:")
print(alphas[alphas > 0])
#print("支持向量个数:")
#print(len((alphas[alphas > 0])[0]))
print("支持向量数据点:")
for i in range(100): 
	if alphas[i] > 0.0: print(dataArr[i],labelArr[i])
'''
lenData=len(dataArr)
lenLabel=len(labelArr)
if lenData==lenLabel:
	print('there are %d items in the file,labelArr is printed:' % lenData)
print(labelArr)


dataMatIn,classLabels=svmMLiA.loadDataSet('testSet.txt')
dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
m,n = shape(dataMatrix)
alphas = mat(zeros((m,1)))

rs1=multiply(alphas,labelMat).T  #alphas,labelMat对应位置元素相乘,结果转置
print(rs1)
print(shape(rs1))

rs2=dataMatrix*dataMatrix[1,:].T
print(rs2)
print(shape(rs2))

print(dataMatrix)
print(dataMatrix[1,:])

A=mat([[1,2,3],[4,5,6]])
B=mat([[1,2,3],[4,5,6]])
C=A*B[1,:].T
print(C)

alphaIold = alphas[66].copy()
print(alphas[66])
print(alphaIold)
'''

