import svmMLiA
from numpy import *

'''
dataArr,labelArr=svmMLiA.loadDataSet('testSet.txt')  #测试文本共100条数据，类别标签为+1和-1两类
lenData=len(dataArr)
lenLabel=len(labelArr)
if lenData==lenLabel:
	print('there are %d items in the file,labelArr is printed:' % lenData)
print(labelArr)
'''

dataMatIn,classLabels=svmMLiA.loadDataSet('testSet.txt')
dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
m,n = shape(dataMatrix)
alphas = mat(zeros((m,1)))

rs1=multiply(alphas,labelMat).T

print(rs1)
print(shape(rs1))