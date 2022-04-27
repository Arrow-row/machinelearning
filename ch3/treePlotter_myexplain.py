'''
Created on Oct 14, 2010

@author: Peter Harrington
'''
import matplotlib.pyplot as plt  #Matplotlib是Python的绘图库,可绘制静态,动态,交互式的图表;Pyplot是Matplotlib的子库,提供了和MATLAB类似的绘图API

#树节点格式常量
decisionNode = dict(boxstyle="sawtooth", fc="0.8")  #定义文本框
leafNode = dict(boxstyle="round4", fc="0.8")  #定义文本框
arrow_args = dict(arrowstyle="<-")  #定义箭头格式

def getNumLeafs(myTree):  #遍历整棵树获取叶节点总数，以确定x轴长度
    numLeafs = 0
    firstStr = list(myTree)[0]  #取出字典myTree第一层的key存于firstStr
    secondDict = myTree[firstStr]  #取出字典第一层key对应的value(也是字典)存于secondDict
    for key in secondDict.keys():  #遍历第二层字典的key
        if type(secondDict[key]).__name__ == 'dict':  #取出第二层字典key对应的value值，测试其数据类型若仍是字典'dict'，则当前节点是判断节点，需要递归调用getNumLeafs()获取其下层字典的叶节点  
            numLeafs += getNumLeafs(secondDict[key])
        else: numLeafs += 1  #否则当前节点是叶节点，numLeafs直接+1
    return numLeafs  #返回叶节点个数

def getTreeDepth(myTree):  #获取树的层数，以确定y轴长度
    maxDepth = 0
    firstStr = list(myTree)[0]  #取出字典myTree第一层的key存于firstStr
    secondDict = myTree[firstStr]  #取出字典第一层key对应的value(也是字典)存于secondDict
    for key in secondDict.keys():  #遍历第二层字典的key
        if type(secondDict[key]).__name__ == 'dict':  #取出第二层字典key对应的value值，测试其数据类型若仍是字典'dict'，递归调用getTreeDepth()获取其下层树高 #test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])  #当前层高=1+下层高度。到达叶子节点，则从递归调用中返回
        else: thisDepth = 1  #否则当前层高=1
        if thisDepth > maxDepth: maxDepth = thisDepth  
    return maxDepth  #返回最大层高

def plotNode(nodeTxt, centerPt, parentPt, nodeType):  #执行实际绘图功能,绘制节点,nodeTxt为要显示的文本,centerPt为文本中心点， parentPt为指向文本的箭头起点, nodeType为箭头所在的点
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',  #全局变量createPlot.ax1定义绘图区;xy=parentPt:箭头终点坐标为parentPt;xycoords='axes fraction':xy坐标系指定为'axes fraction' (0,0是轴域左下角,1,1是右上角)
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)
'''
xy=(横坐标，纵坐标)  #箭头终点
xytext=(横坐标，纵坐标)  #文字的坐标，指的是最左边的坐标
xycoords='str' 、textcoords='str'  #指定xy和xytext的坐标系,有多种可以选择的其他坐标系,如：'axes fraction' ： 0,0 是轴域左下角，1,1 是右上角; 'data'：使用轴域数据坐标系 
va="center",ha="center"  #带边框的文本注释中，表示边框的中心点与文本中心点重合
bbox=nodeType  #文本边框格式，见文件开头节点格式定义
arrowprops=arrow_args  #箭头格式，见文件开头箭头格式定义

'''

def plotMidText(cntrPt, parentPt, txtString):  #计算父节点和子节点的中间位置(xMid,yMid)，在该处添加文本标签信息txtString
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)  #rotation=30表示文案旋转30°

def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #x轴范围
    depth = getTreeDepth(myTree)  #y轴范围
    firstStr = list(myTree)[0]     #非叶节点名称取字典key值
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key], cntrPt, str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')  #调用figure()函数创建画布,参数1为名称或标号,facecolor='white'表示背景色为白色
    fig.clf()  #清空画布
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))  #
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

#def createPlot():
#    fig = plt.figure(1, facecolor='white')  #调用figure()函数绘制图像,参数1表示图号,facecolor='white'表示背景色为白色
#    fig.clf()  #清除当前图像
#    createPlot.ax1 = plt.subplot(111, frameon=False) #ax1是在createPlot函数中定义的全局变量,接受subplot()返回的子图的坐标区;subplot()函数用于在同一画布上绘制多个子图,111表示画布只有一行一列,生成一个图,frameon表示是否绘制坐标轴矩形,这里设置为否
#    #createPlot.ax1=plt.subplot(111)  #这样绘制出的图像有坐标轴矩形
#    #ax1=plt.subplot(111,frameon=False)  #这样定义的不是全局变量,会报错 createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',AttributeError: 'function' object has no attribute 'ax1'
#    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)  #a decision node为注释文本，(0.5, 0.1)为文本中心点，(0.1, 0.5)为箭头起点，decisionNode为文本框格式，见文件开头节点格式定义
#    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
#    plt.show()  #展示画布

def retrieveTree(i):  #存储创建的树信息，便于测试
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

#createPlot(thisTree)
