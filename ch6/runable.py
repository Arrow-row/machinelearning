import svmMLiA

dataArr,labelArr=svmMLiA.loadDataSet('testSet.txt')  #测试文本共100条数据，类别标签为+1和-1两类
lenData=len(dataArr)
lenLabel=len(labelArr)
if lenData==lenLabel:
	print('there are %d items in the file,labelArr is printed:' % lenData)
print(labelArr)