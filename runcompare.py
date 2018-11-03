import time
import csv
import pandas as pd
import random

from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier 

import incmp_plus
from msplit import Msplit
import hhkutil
import incomp
import impute

# clf2 = neighbors.KNeighborsClassifier(n_neighbors = 3, weights = 'distance')
clf2 = LogisticRegression(C=2.0, tol = 0.00005, max_iter = 800)
# mlp = MLPClassifier(tol = 0.0003, max_iter = 800)
clfs1 = [
	incmp_plus.Our_withtree2(clf2, select_limit = 7, beta = 0.2),
	incmp_plus.Our_withtree2(clf2, select_limit = 7, beta = 0.4),
	incmp_plus.Our_withtree2(clf2, select_limit = 7, beta = 0.8),
	incmp_plus.Our_withtree2(clf2, select_limit = 7, beta = 1.2),
	incmp_plus.Our_withtree2(clf2, select_limit = 7, beta = 1.6),
	incmp_plus.Our_withtree2(clf2, select_limit = 7, beta = 2.4),
	incmp_plus.Our_withtree2(clf2, select_limit = 7, beta = 3.2),
	incmp_plus.Our_withtree2(clf2, select_limit = 7, beta = 4.2)
]

rdfile = Msplit(miss_rate = 0.3)

fileTags = ['breast', 'car', 'ecoli', 'iris']
ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

ans = open('ans.txt', 'w')
dic = dict()
imp = impute.impute()
for fileTag in fileTags:
	filename = "new-" + fileTag + ".txt"
	for ratio in ratios:
		rdfile = Msplit(miss_rate = ratio)
		tppre = 0
		tprec = 0
		for itert in xrange(5):
			rdfile.rdfile(filename, 0)
			trainX, trainY = rdfile.get_train(0)
			testX, testY = rdfile.get_test(0)
			imp.read_mtrx(trainX)
			imp.mean_impute()
			imp.mean_impute_mtrx(trainX)
			imp.mean_impute_mtrx(testX)
			clf2.fit(trainX, trainY)
			res = clf2.predict(testX)
			pre, rec = hhkutil.calres(res, testY, 1)
			tppre += pre
			tprec += rec
		tppre /= 5
		tprec /= 5
		print tppre, tprec
		dic[(fileTag, ratio)] = (tppre, tprec)

for fileTag in fileTags:
	ans.write(fileTag+'\n')
	ans.write("p\n")
	for ratio in ratios:
		ans.write(str(ratio) + ' ' + str(dic[(fileTag, ratio)][0]) + '\n')
	ans.write("r\n")
	for ratio in ratios:
		ans.write(str(ratio) + ' ' + str(dic[(fileTag, ratio)][1]) + '\n')