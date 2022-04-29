import bayes
#import bayes_origin

'''
#4.5.2 训练算法：从词向量计算概率
listOPosts,listClasses=bayes.loadDataSet()
myVocabList=bayes.createVocabList(listOPosts)
trainMat=[]
for postinDoc in listOPosts:
	trainMat.append(bayes.setOfWords2Vec(myVocabList,postinDoc))
p0V,p1V,pAb=bayes.trainNB0(trainMat,listClasses)
print(pAb)
print('\n')
print(p0V)
print('\n')
print(p1V)
print('\n')
'''

emailText=open(r'email/ham/6.txt').read()
emailTextSplited=bayes.textParse(emailText)
print(emailText)
print('\n')
print(emailTextSplited)