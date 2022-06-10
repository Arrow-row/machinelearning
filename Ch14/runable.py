import svdRec
from numpy import *

"""
#14.3
Data=svdRec.loadExData()
U,Sigma,VT=linalg.svd(Data)
print('Sigma: ',Sigma)
Sig3=mat([[Sigma[0],0,0],[0,Sigma[0],0],[0,0,Sigma[0]]]) #选择Sigma矩阵中前3个奇异值构建Sig3
dataRecon=U[:,:3]*Sig3*VT[:3,:]
print('dataRecon:',dataRecon)


#14.4.1 相似度计算
myMat=mat(svdRec.loadExData())
print('myMat:\n',myMat)
#欧氏距离
ecludSim1=svdRec.ecludSim(myMat[:,0],myMat[:,4]) 
ecludSim2=svdRec.ecludSim(myMat[:,0],myMat[:,0]) 
print('feat0 to feat4,ecludSim:',ecludSim1)
print('feat0 to feat0,ecludSim:',ecludSim2)
#余弦相似度
cosSim1=svdRec.cosSim(myMat[:,0],myMat[:,4]) 
cosSim2=svdRec.cosSim(myMat[:,0],myMat[:,0]) 
print('feat0 to feat4,cosSim:',cosSim1)
print('feat0 to feat0,cosSim:',cosSim2)
#皮尔逊相关系数
pearsSim1=svdRec.pearsSim(myMat[:,0],myMat[:,4]) 
pearsSim2=svdRec.pearsSim(myMat[:,0],myMat[:,0]) 
print('feat0 to feat4,pearsSim:',pearsSim1)
print('feat0 to feat0,pearsSim:',pearsSim2)

#14.5.1
myMat=mat(svdRec.loadExData())
myMat[0,1]=myMat[0,0]=myMat[2,0]=4
myMat[3,3]=2
print('myMat:\n',myMat)
#对用户2做物品推荐
defaultRec=svdRec.recommend(myMat,2) #默认推荐，距离计算使用余弦相似度。对物品2的预测评分值为2.5，对物品1的预测评分值为2.05
ecludSimRec=svdRec.recommend(myMat,2,simMeas=svdRec.ecludSim) #距离计算使用欧式距离
pearsSimRec=svdRec.recommend(myMat,2,simMeas=svdRec.pearsSim) #距离计算使用皮尔逊相关系数
print('recommendation for user 2:','\ndefaultRec:',defaultRec,'\necludSimRec:',ecludSimRec,'\npearsSimRec:',pearsSimRec)
"""

#14.5.1
myMat=mat(svdRec.loadExData2())
print('myMat:\n',myMat)
'''
print('myMat:\n',type(myMat),'\n',myMat)
print('myMat[:4]:\n',myMat[:4])
print('eye(4)*myMat[:4]:\n',mat(eye(4)*myMat[:4]))
'''
#对用户1做物品推荐
svdEst_cosSim_Rec=svdRec.recommend(myMat,1,estMethod=svdRec.svdEst) #距离计算使用余弦相似度,评分预测方式使用svdEst
svdEst_pearsSim_Rec=svdRec.recommend(myMat,1,simMeas=svdRec.pearsSim,estMethod=svdRec.svdEst) #距离计算使用皮尔逊相关系数,评分预测方式使用svdEst
print('recommendation for user 1:','\nsvdEst_cosSim_Rec:',svdEst_cosSim_Rec,'\nsvdEst_pearsSim_Rec:',svdEst_pearsSim_Rec)

#14.6
svdRec.imgCompress(2)