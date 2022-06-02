'''
Created on Mar 24, 2011
Ch 11 code
@author: Peter
'''
from numpy import *

def loadDataSet(): #创建测试数据集
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet): #构建包含1个项集的集合
    C1 = [] #初始化
    for transaction in dataSet: #遍历dataSet中的子集
        for item in transaction: #遍历子集中元素
            if not [item] in C1: #子集中有而C1没有的元素，将该元素以列表形式添加到列表C1
                C1.append([item])         
    C1.sort()  #C1元素排序
    return list(map(frozenset, C1)) #使用 frozenset 使得集合元素不可变。frozenset() 返回一个冻结的集合，冻结后集合不能再添加或删除任何元素

def scanD(D, Ck, minSupport): #根据最小支持度minSupport，计算Ck在D上的频繁项集。参数：数据集D、候选项集Ck、最小支持度minSupport
    ssCnt = {} #创建空字典
    for tid in D: #遍历D中事件tid，计算候选集Ck的出现次数
        for can in Ck: #遍历候选项集
            if can.issubset(tid): #如果候选项can是事件tid的项集，将can加入字典ssCnt，支持度+1
                if can not in ssCnt: ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems = float(len(D)) #计算D中事件数目
    retList = [] #频繁项集列表，存储频繁项集
    supportData = {} #所有项集支持度字典，存储所有项集及其支持度
    for key in ssCnt: #遍历候选项集字典，
        support = ssCnt[key]/numItems #计算候选项集key的支持度
        if support >= minSupport: #频繁项集加入retList
            retList.insert(0,key) #项集key插入列表retList。insert() 函数用于将指定对象插入列表的指定位置
        supportData[key] = support #项集key及其支持度存入字典supportData
    return retList, supportData #返回频繁项集列表和所有项集支持度字典

def aprioriGen(Lk, k): #此函数在频繁项集Lk上创建Ck项集。输入参数：频繁项集列表Lk，项集元素个数k    creates Ck
    retList = [] #初试化输出项集列表,即Ck项集。函数以{0}、{1}、{2}作为输入，会生成{0,1}、{0,2}以及{1,2}
    lenLk = len(Lk) #获取Lk中频繁项集数目
    for i in range(lenLk): #用子项集Lk[i]与Lk[j]组成更大的项集。为减少计算量，仅当Lk[i]与Lk[j]的前k-2个项相同时，将这两个项集合并
        for j in range(i+1, lenLk):  #循环次数 n*(n-1)/2
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2] 
            #print(k)
            #print(L1)
            #print(L2)
            L1.sort(); L2.sort()
            #print(L1)
            #print(L2)
            if L1==L2: #if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j]) #set union
    return retList #返回Ck项集列表

def apriori(dataSet, minSupport = 0.5): #Apriori算法实现。
    C1 = createC1(dataSet) #创建数据集的1项集
    D = list(map(set, dataSet)) #构建集合表示的数据集D
    L1, supportData = scanD(D, C1, minSupport) #构建1项集的频繁项集和项集支持度 L1, supportData
    L = [L1]
    '''
    dataSet: [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    C1: [frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5})]
    D: [{1, 3, 4}, {2, 3, 5}, {1, 2, 3, 5}, {2, 5}]
    L1: [frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})]
    L: [[frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})]]
    '''
    k = 2
    while (len(L[k-2]) > 0): #Lk为空时退出循环
        Ck = aprioriGen(L[k-2], k)
        print('C%d' %k,Ck)
        Lk, supK = scanD(D, Ck, minSupport) #调用scanD()计算Ck在D上的频繁项集Lk   scan DB to get Lk
        supportData.update(supK) #将supK的项集对应支持度更新到supportData中。字典 update() 函数把字典参数 dict2 的 key/value(键/值) 对更新(添加)到字典 dict1 里
        L.append(Lk) #频繁项集追加到L
        k += 1 #项集大小k+1
    return L, supportData #L为各个k大小的频繁项集L1L2...Lk组成的列表，supportData为全部项集对应的支持度字典

def generateRules(L, supportData, minConf=0.7):  #关联规则生成函数。输入参数：频繁项集L，各个k项集支持度supportData，默认最小置信度minConf      supportData is a dict coming from scanD
    bigRuleList = []
    for i in range(1, len(L)):#i最小值为1，表明从2项集列表开始获取元素   only get the sets with two or more items
        for freqSet in L[i]: #对L[i]中每个频繁项集freqSet创建只包含单个元素集合的列表H1。L[1]=[frozenset({2, 3}), frozenset({3, 5}), frozenset({2, 5}), frozenset({1, 3})]，freqSet=frozenset({2, 3})
            H1 = [frozenset([item]) for item in freqSet] #若freqSet={2, 3}，则 H1=[{2},{3}]
            #print('H1:',H1)
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf) #调用calcConf(), 求出freqSet上满足阈值的规则
    return bigRuleList  #返回满足阈值要求的关联规则及其可信度列表   [(frozenset({5}), frozenset({2}), 1.0), (frozenset({2}), frozenset({5}), 1.0), (frozenset({1}), frozenset({3}), 1.0)]        

def calcConf(freqSet, H, supportData, brl, minConf=0.7): #计算关联规则的可信度。输入参数：
    prunedH = [] #列表prunedH用于返回       create new list to return
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #计算L[1]中各个频繁项集freqSet上关联规则的可信度conf       calc confidence
        if conf >= minConf: #若当前规则可信度conf大于阈值，将规则和对应可信度打印并以元组形式添加到列表brl
            print(freqSet-conseq,'-->',conseq,'conf:',conf) #frozenset({5}) --> frozenset({2}) conf: 1.0
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq) 
            #print('prunedH:',prunedH) #prunedH: [frozenset({2}), frozenset({5})]
    return prunedH #返回满足最小可信度的规则组成元素列表

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7): 
    m = len(H[0]) 
    print('H:',H)
    print('m',m)
    if (len(freqSet) > (m + 1)): #try further merging
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
            
def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print(itemMeaning[item])
        print("           -------->")
        for item in ruleTup[1]:
            print(itemMeaning[item])
        print("confidence: %f" % ruleTup[2])
        print()       #print a blank line
        
"""         
from time import sleep
from votesmart import votesmart
votesmart.apikey = 'get your api key first'
def getActionIds():
    actionIdList = []; billTitleList = []
    fr = open('recent20bills.txt') 
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum) #api call
            for action in billDetail.actions:
                if action.level == 'House' and \
                (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print('bill: %d has actionId: %d' % (billNum, actionId))
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print("problem getting bill %d" % billNum)
        sleep(1)                                      #delay to be polite
    return actionIdList, billTitleList
        
def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
    for billTitle in billTitleList:#fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}#list of items in each transaction (politician) 
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print('getting votes for actionId: %d' % actionId)
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName): 
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except: 
            print("problem getting actionId: %d" % actionId)
        voteCount += 2
    return transDict, itemMeaning
"""