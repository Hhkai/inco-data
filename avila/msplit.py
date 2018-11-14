import random

class Msplit:
	def __init__(self, split_n = 0.3, miss_rate = 0.3, sample_rate = 1.0, feature_rate = 1.0):
		self.split_n = split_n # the ratio of test sample
		self.dataX = dict()
		self.dataY = dict()
		self.miss_rate = miss_rate
		self.sample_rate = sample_rate
		self.feature_rate = feature_rate
	def rdfile(self, fileName, fileAtt):
		fr = open(fileName,'r')
		testX = []
		lable = []
		lines = fr.readlines()
		# print fileName
		rows = range(len(lines))
		if self.sample_rate < 1.0:
			rows = random.sample(rows, int(len(lines) * self.sample_rate))
		for ind, line in enumerate(lines):
			if ind not in rows:
				continue
			lineNew = line.strip().split(',')
			for i in xrange(len(lineNew)):
				try:
					lineNew[i] = float(lineNew[i])
				except:
					lineNew[i] = '?'
			testX.append(lineNew[:-1])
			lable.append(int(lineNew[-1]))
		fr.close()
		if self.miss_rate > 0: # make more missing
			total = len(testX) * len(testX[0])
			miss = random.sample(range(total), int(total * self.miss_rate))
			for i in miss:
				testX[i / len(testX[0])][i % len(testX[0])] = '?'
		self.dataX[fileAtt] = testX
		self.dataY[fileAtt] = lable
		if fileAtt == 0:
			self.rdsample(lable)
		print "read end , row = ", len(testX)
	def rdsample(self, dataY):
		dict_y = dict()
		for ind, y in enumerate(dataY):
			if dict_y.has_key(y) == False:
				dict_y[y] = set()
			dict_y[y].add(ind)
		#
		testsample = dict()
		trainsample = dict()
		for y in dict_y:
			cury_n = len(dict_y[y])
			testsample[y] = set(random.sample(dict_y[y], int(cury_n * self.split_n)))
			trainsample[y] = dict_y[y] - testsample[y]
		self.testsample = testsample
		self.trainsample = trainsample
	def get_test(self, fileAtt):
		resX = []
		resY = []
		for y in self.testsample:
			for ind in self.testsample[y]:
				resX.append(self.dataX[fileAtt][ind])
				resY.append(self.dataY[fileAtt][ind])
		return resX, resY
	def get_train(self, fileAtt):
		resX = []
		resY = []
		for y in self.trainsample:
			for ind in self.trainsample[y]:
				resX.append(self.dataX[fileAtt][ind])
				resY.append(self.dataY[fileAtt][ind])
		return resX, resY
