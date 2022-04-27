#示例3.4图像3-8

import treePlotter
import trees

fr=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels=['age','prescript','astgmatic','tearRate']
lensesTree=trees.createTree(lenses,lensesLabels)
treePlotter.createPlot(lensesTree)
