import numpy as np
import operator

def linesInText(filename):  #计算文本文件中内容行数
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)
    return numberOfLines