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


# clf2 = neighbors.KNeighborsClassifier(n_neighbors = 3, weights = 'distance')
clf2 = LogisticRegression(C=2.0, tol = 0.00005, max_iter = 800)
# mlp = MLPClassifier(tol = 0.0003, max_iter = 800)
clfs1 = incmp_plus.Our_withtree2(clf2, select_limit = 7, beta = 1.0)

#treeC = incomp.IN_cmp2(clf2)
#annC = incomp.IN_cmp1(debug = 0)

rdfiles = [
	Msplit(miss_rate = 0.3, sample_rate = 0.5),
	Msplit(miss_rate = 0.3, sample_rate = 0.6),
	Msplit(sample_rate = 0.7),
	Msplit(sample_rate = 0.8),
	Msplit(sample_rate = 0.9)
]

fileTag = "ecoli2"
filename = "new-" + fileTag + ".txt"
# filename1 = "new-" + fileTag + ".txt-" + ratio + ".txt"

a=[]

for itert in xrange(5):

	row = []
	for i in rdfiles:
		i.rdfile(filename, 0)
		trainX, trainY = i.get_train(0)
		testX, testY = i.get_test(0)
		clfs1.fit(trainX, trainY)
		res = clfs1.predict(testX)
		pre, rec = hhkutil.calres(res, testY, 1)
		#res = i.predict(testX2)
		#pre, rec = hhkutil.calres(res, testY2, 1)
		row.append(pre)
		row.append(rec)
		row.append(' ')
	a.append(row)

print "===="

out = open('tsr-' + fileTag + '.csv', 'w')
csv_writer = csv.writer(out)

for i in a :
	csv_writer.writerow(i)
