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

ourtree = incmp_plus.Our_withtree2(clf2, select_limit = 7, beta = 1.0)

histree = incomp.IN_cmp2(clf2)

# fileTags = ['breast', 'car', 'ecoli', 'iris']
ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

ans = open('ans.txt', 'w')
dic = dict()
imp = impute.impute()

tsf = "avila-ts-01.txt"
trf = "avila-tr-01.txt"

for ratio in ratios:
	tr_data = Msplit(miss_rate = ratio, split_n = 0)
	ts_data = Msplit(miss_rate = ratio, split_n = 1)
	tppre = 0
	tprec = 0
	for itert in xrange(5):
		tr_data.rdfile(trf, 0)
		ts_data.rdfile(tsf, 0)
		trainX, trainY = tr_data.get_train(0)
		testX, testY = ts_data.get_test(0)
		####
		ourtree.fit(trainX, trainY)
		res = ourtree.predict(testX)
		pre, rec = hhkutil.calres(res, testY, 1)

		#imp.read_mtrx(trainX)
		#imp.mean_impute()
		#imp.mean_impute_mtrx(trainX)
		#imp.mean_impute_mtrx(testX)
		#clf2.fit(trainX, trainY)
		#res = clf2.predict(testX)
		#pre, rec = hhkutil.calres(res, testY, 1)
		tppre += pre
		tprec += rec
	tppre /= 5
	tprec /= 5
	print tppre, tprec
	dic[ratio] = (tppre, tprec)

for ratio in ratios:
	ans.write(str(ratio) + ' ' + str(dic[ratio][0]) + '\n')
ans.write("r\n")
for ratio in ratios:
	ans.write(str(ratio) + ' ' + str(dic[ratio][1]) + '\n')
ans.close()