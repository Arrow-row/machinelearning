from numpy import *
import kNN
import matplotlib
import matplotlib.pyplot as plt

'''
datingTestSet.txt 中为样本数据集，前3列为样本特征，第4列为类型标签
第1列:每年获得的飞行常客里程数
第2列:玩视频游戏所耗时间百分比
第3列:每周消费的冰淇淋公升数
第4列:类型标签，包括不喜欢、一般、很喜欢3类,分别用数字1、2、3标识
'''
datingDataMat,datingLabels=kNN.file2matrix('testdemo.txt') 
normMat,ranges,minVals=kNN.autoNorm(datingDataMat)
print('normMat：\n',normMat,'\n','ranges：\n',ranges,'\n','minVals:\n',minVals)
fig=plt.figure()
ax=fig.add_subplot(111)

'''
#fig_2-3
ax.scatter(datingDataMat[:,1],datingDataMat[:,2])  
plt.xlabel('Percentage of Time Spent Playing Video Games')
plt.ylabel('Liters of Ice Cream Consumed Per Week')
'''

'''
#fig_2-4
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))  
plt.xlabel('Percentage of Time Spent Playing Video Games')
plt.ylabel('Liters of Ice Cream Consumed Per Week')
'''

#fig_2-5
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))  
plt.xlabel('Frequent Flyier Miles Earned Per Year')
plt.ylabel('Percentage of Time Spent Playing Video Games')
plt.show()

