#import bayes
import bayes_origin


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