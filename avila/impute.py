import copy
import numpy as np
from sklearn import neighbors

class impute:
	def __init__(self):
	    self.mtrx = []
	def read_mtrx(self, mtrx):
		self.mtrx = copy.deepcopy(mtrx)
		self.col_n = len(mtrx[0])
		self.row_n = len(mtrx)
	def mean_impute(self):
		col_sum = [0] * self.col_n
		col_cnt = [0] * self.col_n
		for row in self.mtrx:
			for col_id, col in enumerate(row):
				try:
					_ = float(col)
					col_sum[col_id] += _ 
					col_cnt[col_id] += 1
				except:
					pass
		self.col_means = [col_sum[col_id] * 1.0 / col_cnt[col_id] for col_id in xrange(self.col_n)]
		#
		for row in self.mtrx:
			for col_id, col in enumerate(row):
				try:
					_ = float(col)
				except:
					row[col_id] = self.col_means[col_id]
		#
	def KNNI(self):
		self.complete = []
		for row in self.mtrx:
			is_complete = True
			for i in row:
				try:
					_ = float(i)
				except:
					is_complete = False
					break
			if is_complete:
				self.complete.append(copy.deepcopy(row))
		#
		complete_sample_n = len(self.complete)
		if complete_sample_n < 3:
			complete_sample_n = 1
			print "complete_sample_n < 3"
		else:
			complete_sample_n = 3
		#
		regressors = dict()
		complete_T = np.array(self.complete).T
		# knn = neighbors.KNeighborsRegressor(n_neighbors = 3, weights = 'distance')
		for row_id, row in enumerate(self.mtrx):
			miss = []
			temp = []
			for col_id, col in enumerate(row):
				try:
					_ = float(col)
					temp.append(_)
				except:
					miss.append(col_id)
			if len(miss) != 0:
				miss = tuple(miss)
				if regressors.has_key(miss) == False:
					train_x = []
					train_y = []
					for i in xrange(self.col_n):
						if i in miss:
							train_y.append(complete_T[i])
						else:
							train_x.append(complete_T[i])
					#
					regressors[miss] = neighbors.KNeighborsRegressor(n_neighbors = complete_sample_n)
					train_x = np.array(train_x).T 
					train_y = np.array(train_y).T 
					regressors[miss].fit(train_x, train_y)
				#
				regress_y = regressors[miss].predict([temp])[0]
				for ans_id, col_id in enumerate(miss):
					self.mtrx[row_id][col_id] = regress_y[ans_id]
		#
	def mean_impute_mtrx(self, X):
		for row in X:
			for col_id, col in enumerate(row):
				try:
					_ = float(col)
				except:
					row[col_id] = self.col_means[col_id]
#
if __name__ == "__main__":
	a = [
		 [1, 2, 3],
		 [4, 5, 6],
		 [7, 8, '?']
		]
	cl = impute()
	cl.read_mtrx(a)
	cl.KNNI()
	print cl.mtrx