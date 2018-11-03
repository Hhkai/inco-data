import copy
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
	cl.mean_impute()
	print cl.col_means
	print cl.mtrx