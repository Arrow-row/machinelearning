import apriori
from numpy import *

#11.3.1
dataSet=apriori.loadDataSet() #创建测试数据集
print('dataSet:',dataSet)
C1=apriori.createC1(dataSet) #构建一项集C1
print('C1:',C1)
D=list(map(set,dataSet)) #构建集合表示的数据集D
print('D:',D)
L1,suppData0=apriori.scanD(D,C1,0.5) #0.5作为最小支持度，获取C1中的频繁项集
L=[L1]
print('L1:',len(L1),L1)
print('L:',L)
print('suppData0:',suppData0)
'''
dataSet: [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
C1: [frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5})]
D: [{1, 3, 4}, {2, 3, 5}, {1, 2, 3, 5}, {2, 5}]
L1: [frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})]
L: [[frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})]]
suppData0: {frozenset({1}): 0.5, frozenset({3}): 0.75, frozenset({4}): 0.25, frozenset({2}): 0.75, frozenset({5}): 0.75}
'''
#11.3.2
L,suppData=apriori.apriori(dataSet)
print('L:',L)
print('suppData:',suppData)

#11.4
L, suppData = apriori.apriori(dataSet, minSupport=0.5)
rules = apriori.generateRules(L, suppData, minConf=0.7)
print(rules)

