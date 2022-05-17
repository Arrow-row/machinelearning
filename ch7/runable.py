import adaboost
from numpy import *

'''datMat,classLabels = adaboost.loadSimpData()    #<class 'numpy.matrixlib.defmatrix.matrix'>   <class 'list'>
classifierArray=adaboost.adaBoostTrainDS(datMat,classLabels,9)
print(classifierArray)
res=adaboost.adaClassify([0,0],classifierArray) #待分类样本是[0,0]，res存放返回的分类结果
print("the ultimate classification result:",res)
res=adaboost.adaClassify([[5,5],[0,0]],classifierArray) #待分类样本是[5,5],[0,0]，res存放返回的分类结果
print("the ultimate classification result:",res)
'''
numIt=[1,10,50,100,500,1000]
for i in range(0,len(numIt)):
	print("numIt = ",numIt[i])
	datArr,labelArr=adaboost.loadDataSet('horseColicTraining2.txt')
	classifierArray=adaboost.adaBoostTrainDS(datArr,labelArr,numIt[i])
	testArr,testLabelArr=adaboost.loadDataSet('horseColicTest2.txt')
	prediction10=adaboost.adaClassify(testArr,classifierArray)
	errArr=mat(ones((67,1)))
	errorRate=errArr[prediction10 != mat(testLabelArr).T].sum()/67
	print("prediction errorRate: %.2f" %errorRate)


'''
aggClassEst = mat(zeros((5,1)))
aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((5,1)))
print(mat(classLabels).T)
print(sign(aggClassEst) != mat(classLabels).T)
print(aggErrors)

[[ 1.]    
 [ 1.]    
 [-1.]    
 [-1.]    
 [ 1.]]   
[[ True]  
 [ True]  
 [ True]  
 [ True]  
 [ True]] 
[[1.]     
 [1.]     
 [1.]     
 [1.]     
 [1.]]    
'''


'''
D=mat(ones((5,1))/5)
bestStump,minError,bestClasEst=adaboost.buildStump(datMat,classLabels,D)
print("bestStump:",bestStump,"\nminError:",minError,"\nbestClasEst:",bestClasEst)
'''