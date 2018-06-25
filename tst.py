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


rdfile = Msplit()

fileTag = "breast"
ratio = "0.3"
filename = "new-" + fileTag + ".txt"
filename1 = "new-" + fileTag + ".txt-" + ratio + ".txt"

rdfile.rdfile(filename1, 0)  # train
rdfile.rdfile(filename, 1)  # test, no miss 

trainX, trainY = rdfile.get_train(0)
testX, testY = rdfile.get_test(0)
testX2, testY2 = rdfile.get_test(1)
trainX2, trainY2 = rdfile.get_train(1)

clf = LogisticRegression(C=1.5)
cmp4 = incmp_plus.Our_withtree1(clf)

time2 = time.clock()
cmp4.fit(trainX, trainY)
res2 = cmp4.predict(testX2)
time2 = time.clock() - time2
pre2, rec2 = hhkutil.calres(res2, testY2, 1)

print "res:", pre2, rec2
leny = len(testY2)
random_ans = []
for i in xrange(leny):
	x = random.random()
	y = 0 if x >= 0.5 else 1
	random_ans.append(y)

pre3, rec3 = hhkutil.calres(random_ans, testY, 1)
print pre3, rec3
exit()


# clf2 = neighbors.KNeighborsClassifier(n_neighbors = 3, weights = 'distance')
# clf = neighbors.KNeighborsClassifier(n_neighbors = 5)
clf2 = LogisticRegression(C=2.0, tol = 0.00005, max_iter = 800)
mlp = MLPClassifier(tol = 0.0003, max_iter = 800)
# cmp1 = incomp.IncomNoselect(clf1, weight = 'same')
cmp2 = incmp_plus.Our_withtree1(clf2)
cmp3 = incmp_plus.Our_withtree2(clf2)
# cmp3 = incomp.IncomWeighed(clf, alpha = 0.8)
# cmp4 = incomp.IN_cmp1(debug = 1)
# cmp5 = incomp.IN_cmp2(clf)
# cmp6 = incomp.IN_cmp3(clf)

fileTag = "breast"
ratio = "0.5"
filename = "new-" + fileTag + ".txt"
filename1 = "new-" + fileTag + ".txt-" + ratio + ".txt"

a=[]

for itert in xrange(3):
	rdfile.rdfile(filename1, 0)  # train
	rdfile.rdfile(filename, 1)  # test, no miss 

	trainX, trainY = rdfile.get_train(0)
	testX, testY = rdfile.get_test(0)
	testX2, testY2 = rdfile.get_test(1)
	trainX2, trainY2 = rdfile.get_train(1)


	time1 = time.clock()
	cmp2.fit(trainX, trainY)
	res2 = cmp2.predict(testX2)
	time1 = time.clock() - time1
	pre1, rec1 = hhkutil.calres(res2, testY2, 1)

	time2 = time.clock()
	cmp2.fit(trainX, trainY)
	res2 = cmp2.predict(testX)
	time2 = time.clock() - time2
	pre2, rec2 = hhkutil.calres(res2, testY, 1)

	time3 = time.clock()
	cmp3.fit(trainX, trainY)
	res3 = cmp3.predict(testX2)
	time3 = time.clock() - time3
	pre3, rec3 = hhkutil.calres(res3, testY2, 1)

	time4 = time.clock()
	cmp3.fit(trainX, trainY)
	res3 = cmp3.predict(testX)
	time4 = time.clock() - time4
	pre4, rec4 = hhkutil.calres(res3, testY, 1)

	a.append([pre1, rec1, time1,pre2, rec2, time2,pre3, rec3, time3,pre4, rec4, time4])

print "===="

out = open('z_out.csv', 'w')
csv_writer = csv.writer(out)

for i in a :
	csv_writer.writerow(i)
