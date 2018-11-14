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

rdfile = Msplit(miss_rate = 0.3, sample_rate = 1.0)

filename = 'new-ecoli-0.txt'
# filename1 = "new-" + fileTag + ".txt-" + ratio + ".txt"

a=[]

for itert in xrange(5):

	row = []
	rdfile.rdfile(filename, 0)
	trainX, trainY = rdfile.get_train(0)
	testX, testY = rdfile.get_test(0)
	clfs1.fit(trainX, trainY)
	res = clfs1.predict(testX)
	pre, rec = hhkutil.calres(res, testY, 1)
	row.append(pre)
	row.append(rec)
	row.append(' ')
	a.append(row)

print "===="

out = open('st-' + filename + '.csv', 'w')
csv_writer = csv.writer(out)

for i in a :
	csv_writer.writerow(i)
