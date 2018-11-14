import copy
import time
import numpy as np
import random

from sklearn.neural_network import MLPClassifier 

import hhkutil

class Our_withtree1:

    def __init__(self, clf):
        # string
        self.clf = clf
        self.debug = False

    def fit(self, trainX, trainY):
        self.trainX = trainX
        self.trainY = trainY
        self.row_n = len(trainY)
        self.col_n = len(trainX[0])
        assert self.row_n == len(trainX)

        CDS_tree = dict()

        MS = []
        # MS[i] = (1, 2) 
        # the i-th sample in dataX miss the (1, 2)
        for ind, X in enumerate(self.trainX):
            numb = set()
            for i in xrange(self.col_n):
                if X[i] == None or X[i] == '?': 
                    numb.add(i)
            temp_miss = tuple(numb)
            MS.append(temp_miss)
        # 
        classifier_key = set(MS)
        self.classifier_key = classifier_key
        for i in classifier_key:
            CDS_tree[i] = set()
        for index, i in enumerate(MS):
            for j in classifier_key:
                if set(i).issubset(set(j)):
                    CDS_tree[j].add(index)
        #
        self.CDS_tree = CDS_tree
        self.MS = MS
        #

        #####  make MI
        MIs = []
        for i in xrange(self.col_n): 
            x = []
            y = []
            for j in xrange(self.row_n):
                if self.trainX[j][i] != None and self.trainX[j][i] != '?':
                    x.append(self.trainX[j][i])
                    y.append(self.trainY[j])   
            MIs.append(hhkutil.MI(x, y))
        self.MIs = MIs

        #### make ada_votes
        ada_votes = dict()
        for i in set(self.MS):
            if len(i) == self.col_n:
                continue
            allSame = True
            curYs = tuple(self.CDS_tree[i])
            for y in curYs:
                if self.trainY[y] != self.trainY[curYs[0]]:
                    allSame = False
                    break
            if allSame == True:
                continue
            kaka = Ada_vote(self, i)
            kaka.train_go()
            ada_votes[i] = kaka
        #
        self.ada_votes = ada_votes
    #
    def predict(self, testX):
        res = []
        for curX in testX:
            numb = set()
            for ind, x in enumerate(curX):
                if x == None or x == '?': 
                    numb.add(ind)
            miss = tuple(numb)
            # print "testX miss:", miss
            if self.ada_votes.has_key(miss) == False:
                res.append(-2)
                continue
            else:
                cur_res = self.ada_votes[miss].predict(curX)
                res.append(cur_res)
        return res
#
class Ada_vote:
    def __init__(self, tree, miss):
        self.clf = tree.clf
        self.tree = tree
        self.miss = miss
        # mk miss list
        subsets = hhkutil.subSet(miss)
        mi_sum_dict = dict()
        for i in subsets:
            t_sum = 0
            for j in i:
                t_sum += tree.MIs[j]
            mi_sum_dict[i] = t_sum
        def sum_mi(x):
            return -mi_sum_dict[x]
        a = list(subsets)
        a.sort(key=sum_mi)
        self.miss_list = a 
        # the sort is delete the min MI

        # print self.miss_list

        if len(self.miss_list) == 0:
            self.miss_list.append(tuple())
        #
        # mk train data
        id4xid = dict()
        trainX = []
        trainY = []
        index = 0
        for i in tree.CDS_tree[miss]:
            curX = []
            for j in xrange(self.tree.col_n):
                if j not in miss:
                    curX.append(self.tree.trainX[i][j])
            trainX.append(curX)
            trainY.append(self.tree.trainY[i])
            id4xid[i] = index
            index += 1
        self.trainX = trainX
        self.trainY = trainY
        self.id4xid = id4xid
        # 
        self.stop = 0
        self.last_e = 1
        self.sample_n = len(trainY)
        self.sample_wi = [1] * self.sample_n
        self.classifier = []
        self.classifier_wi = []
        #
        self.trainEnd = False
    #
    def train(self):
        lr = copy.deepcopy(self.clf)
        lr.fit(self.trainX, self.trainY, self.sample_wi)
        self.classifier.append(lr)
        ansY = lr.predict(self.trainX)
        
        cur_e_wi = hhkutil.normaled(self.sample_wi)
        cur_e = 0
        for ind, i in enumerate(ansY):
            if i != self.trainY[ind]:
                cur_e += cur_e_wi[ind]
        if cur_e > self.last_e:
            self.stop += 1
        else:
            self.stop = 0

        # cur_e != 0
        if cur_e == 0:
            print "cur_e = 0"
            cur_e = 0.0001
            self.stop += 9

        self.last_e = cur_e
        alpha_m = 0.5 * np.log((1 - cur_e) / cur_e)
        self.classifier_wi.append(alpha_m)
        #
        for ind, i in enumerate(ansY):
            if i != self.trainY[ind]:
                self.sample_wi[ind] *= np.exp(alpha_m)
            else:
                self.sample_wi[ind] *= np.exp(-alpha_m)
        self.sample_wi = hhkutil.normaled(self.sample_wi, sum_x = 1 * self.sample_n)
    def train_go(self):
        for i in self.miss_list:
            if self.stop > 2:
                break
            if self.tree.CDS_tree.has_key(i) == False:
                continue
            add_data_ids = self.tree.CDS_tree[i]
            tp_add_wi = len(add_data_ids) * 1.0 / self.sample_n
            for j in add_data_ids:
                self.sample_wi[self.id4xid[j]] += tp_add_wi
            #
            self.train()
        #
        # sumwi = sum(self.classifier_wi)
        # print sumwi
        # exit()
        self.classifier_wi = hhkutil.normaled(self.classifier_wi)
        self.trainEnd = True
    #
    def predict(self, testX):
        # mk data
        tstX = []
        for j in xrange(self.tree.col_n):
            if j not in self.miss:
                tstX.append(testX[j])
        #
        y = 0
        for idx_c, i in enumerate(self.classifier):
            cur_y = i.predict([tstX])[0]
            y += self.classifier_wi[idx_c] * cur_y
            # print cur_y, self.classifier_wi[idx_c]
        #
        # print "y", y
        # exit()
        if y >= 0.5:
            return 1
        else:
            return 0
    #
#
class Our_withtree2:
    # in "withtree1", the complete data do not have the 
    # 'my ada' process
    # consider the process in another way, we get 'withtree2'
    def __init__(self, clf, debug = False, select_limit = -1, beta = 1.0):
        # string
        self.clf = clf
        self.debug = debug
        self.select_limit = select_limit
        self.beta = beta

    def fit(self, trainX, trainY):
        self.trainX = trainX
        self.trainY = trainY
        self.row_n = len(trainY)
        self.col_n = len(trainX[0])
        assert self.row_n == len(trainX)

        CDS_tree = dict()

        MS = []
        # MS[i] = (1, 2) 
        # the i-th sample in dataX miss the (1, 2)
        for ind, X in enumerate(self.trainX):
            numb = set()
            for i in xrange(self.col_n):
                if X[i] == None or X[i] == '?': 
                    numb.add(i)
            temp_miss = tuple(numb)
            MS.append(temp_miss)
        # 
        classifier_key = set(MS)
        self.classifier_key = classifier_key
        for i in classifier_key:
            CDS_tree[i] = set()
        for index, i in enumerate(MS):
            for j in classifier_key:
                if set(i).issubset(set(j)):
                    CDS_tree[j].add(index)
        #
        self.CDS_tree = CDS_tree
        self.MS = MS
        #

        #####  make MI
        MIs = []
        for i in xrange(self.col_n): 
            x = []
            y = []
            for j in xrange(self.row_n):
                if self.trainX[j][i] != None and self.trainX[j][i] != '?':
                    x.append(self.trainX[j][i])
                    y.append(self.trainY[j])   
            MIs.append(hhkutil.MI(x, y))
        self.MIs = MIs

        #### make ada_votes
        ada_votes = dict()
        for i in set(self.MS):
            if len(i) == self.col_n:
                continue
            allSame = True
            curYs = tuple(self.CDS_tree[i])
            for y in curYs:
                if self.trainY[y] != self.trainY[curYs[0]]:
                    allSame = False
                    break
            if allSame == True:
                continue
            kaka = Ada_vote2(self, i, select_limit = self.select_limit, beta = self.beta)
            kaka.train_go()
            ada_votes[i] = kaka
        #
        self.ada_votes = ada_votes
    #
    def predict(self, testX):
        res = []
        for curX in testX:
            numb = set()
            for ind, x in enumerate(curX):
                if x == None or x == '?': 
                    numb.add(ind)
            miss = tuple(numb)
            # print "testX miss:", miss
            if self.ada_votes.has_key(miss) == False:
                res.append(-2)
                continue
            else:
                cur_res = self.ada_votes[miss].predict(curX)
                res.append(cur_res)
        return res
class Ada_vote2:
    def __init__(self, tree, miss, select_limit = -1, beta = 1.0):
        self.clf = tree.clf
        self.tree = tree
        self.miss = miss
        self.debug = self.tree.debug
        self.select_limit = select_limit
        self.beta = beta
        # mk miss list
        S = set(range(tree.col_n))
        left_miss_f_set = S - set(miss)
        subsets = hhkutil.subSet(list(left_miss_f_set))

        mi_sum_dict = dict()
        for i in subsets:
            t_sum = 0
            for j in i:
                t_sum += tree.MIs[j]
            mi_sum_dict[i] = t_sum
        def sum_mi(x):
            return mi_sum_dict[x]
        a = list(subsets)
        a.sort(key=sum_mi)

        # print "raw miss", miss
        self.raw_miss = miss
        sets_consider = []
        for i in a[:select_limit]:
            listx = list(set(miss).union(set(i)))
            # print listx
            sets_consider.append(tuple(listx))
        
        self.miss_list = sets_consider 

        ##
        for i in self.miss_list:
            assert set(miss).issubset(set(i))

        if len(a) > 1:
            assert mi_sum_dict[a[0]] < mi_sum_dict[a[1]]
        # print self.miss_list

        if len(a) == 0:
            print "warning:len(a) == 0", miss
        if len(self.miss_list) == 0:
            self.miss_list.append(tuple())
        #
        self.stop = 0
        self.last_e = 1
        self.classifier = dict()
        self.classifier_wi = dict()
        #
        self.trainEnd = False
    #
    def train(self, miss_feature):
        lr = copy.deepcopy(self.clf)
        # make trainX and trainY
        trainX = []
        trainY = []
        for i in self.selected_id:
            x = []
            for j in xrange(self.tree.col_n):
                if j not in miss_feature:
                    x.append(self.tree.trainX[i][j])
            trainX.append(x)
            trainY.append(self.tree.trainY[i])
        # make end
        assert len(trainX) == len(trainY) and len(trainY) == len(self.sample_wi)
        lr.fit(trainX, trainY, self.sample_wi)
        self.classifier[miss_feature] = lr
        ansY = lr.predict(trainX)
        
        cur_e_wi = hhkutil.normaled(self.sample_wi)
        cur_e = 0
        for ind, i in enumerate(ansY):
            if i != trainY[ind]:
                cur_e += cur_e_wi[ind]
        if cur_e > self.last_e:
            self.stop += 1
        else:
            self.stop = 0

        # cur_e != 0
        if cur_e == 0:
            # print "cur_e = 0"
            cur_e = 0.0001
            self.stop += 9

        self.last_e = cur_e
        alpha_m = 0.5 * np.log((1 - cur_e) / cur_e)
        self.classifier_wi[miss_feature] = alpha_m
        #
        for ind, i in enumerate(ansY):
            if i != trainY[ind]:
                self.sample_wi[ind] *= np.exp(self.beta * alpha_m)
            else:
                self.sample_wi[ind] *= np.exp(- self.beta * alpha_m)
        self.sample_wi = self.sample_wi[:self.sample_n]
        self.sample_wi = hhkutil.normaled(self.sample_wi, sum_x = 1 * self.sample_n)
    def train_go(self):
        # every time give a new sample_weight, the idset of data
        # trainX and trainY is calculated in every itre..
        
        self.raw_selected_id = list(self.tree.CDS_tree[self.raw_miss])
        self.sample_n = len(self.raw_selected_id)
        self.sample_wi = [1.0 / self.sample_n] * self.sample_n
        self.selected_id = copy.deepcopy(self.raw_selected_id)
        self.train(self.raw_miss)
        
        for i in self.miss_list:
            #if self.stop > 7:
            #    break
            if self.tree.CDS_tree.has_key(i) == False:
                continue
            if len(i) == self.tree.col_n:
                continue
            self.selected_id = copy.deepcopy(self.raw_selected_id)
            add_data_ids = self.tree.CDS_tree[i]
            tp_add_wi = 1.0 / len(add_data_ids)
            for j in add_data_ids:
                self.selected_id.append(j)
                self.sample_wi.append(tp_add_wi)
            #
            self.train(i)
        #
        # sumwi = sum(self.classifier_wi)
        # print sumwi
        # exit()
        if len(self.classifier_wi) == 0:
            print self.miss_list
            return 
        self.classifier_wi = hhkutil.normaled(self.classifier_wi)
        self.trainEnd = True
    #
    def predict(self, testX):
        y = 0
        for miss_f in self.classifier:
            temX = []
            for j in xrange(self.tree.col_n):
                if j not in miss_f:
                    temX.append(testX[j])
            cur_y = self.classifier[miss_f].predict([temX])[0]
            # cur_y = self.classifier[miss_f].predict_proba([temX])[0][1]
            y += self.classifier_wi[miss_f] * cur_y
            if self.debug:
                print self.classifier_wi[miss_f], '*', cur_y
        #
        if self.debug:
            print "--- y", y
        # print "y", y
        # exit()
        if y >= 0.5:
            return 1
        else:
            return 0
    #
#

if __name__ == "__main__":
    print "main: incmp_plus.py"