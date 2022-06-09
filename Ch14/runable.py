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
"""

#14.4.1 相似度计算
myMat=mat(svdRec.loadExData())
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