import copy
import time
import numpy as np
import random

from sklearn.neural_network import MLPClassifier 

import hhkutil

class IncomNoselect:
    def __init__(self, clsfier):
        self.clsfier = clsfier

    def fit(self, trainX, trainY):
        self.trainX = trainX
        self.trainY = trainY
        self.row_n = len(trainY)
        self.col_n = len(trainX[0])
        assert self.row_n == len(trainX)

        MvS = [] # for every data
        for itt in trainX: 
            numb = set()
            for ind, x in enumerate(itt):
                if x != None and x != '?': 
                    numb.add(ind)
            MvS.append(tuple(numb))
        # dataX[i] is a sample
        # MvS[i] is the no missing features of this sample.
        # eg: MvS[3] = (0, 1, 3)
        MS = tuple(set(MvS)) 
        self.MS = MS  

        CDS = []
        for i in MS:   
            # i is some kind of MvS
            #eg: i = (0, 1)
            CDSi = []
            for ind, mvs in enumerate(MvS):
                if set(i).issubset(set(mvs)): 
                    CDSi.append(ind)
            CDS.append(CDSi)
        self.CDS = CDS
        self.CDS_n = len(CDS)
        # CDS[3] = [id1, id2, id3, ...]

        clsfs = []
        obsolete = set()
        for ind, i in enumerate(CDS):
            fea = self.MS[ind]
            nclsfier = copy.deepcopy(self.clsfier)
            if len(fea) == 0:
                clsfs.append(nclsfier)
                continue
            tempX = []
            tempY = []
            for j in i:
                x = []
                for col in fea:
                    x.append(self.trainX[j][col])
                tempX.append(x)
                tempY.append(self.trainY[j])
            #
            # print fea
            # for x in tempX:
            #     print x
            #
            if len(tempX) <= 12:
                obsolete.add(ind)
                clsfs.append(nclsfier)
                continue
            nclsfier.fit(tempX, tempY)
            clsfs.append(nclsfier)
        self.clsfs = clsfs
        self.obsolete_id = obsolete

        MIs = []
        for i in xrange(self.col_n): 
            x = []
            y = []
            for j in xrange(self.row_n):
                if self.trainX[j][i] != None:
                    x.append(self.trainX[j][i])
                    y.append(self.trainY[j])   
            MIs.append(hhkutil.MI(x, y))
        #
        # print MIs
        CDS_sumSI = []
        for i in self.MS:
            #for every CDS
            temp_sum = 0
            for j in i:
                temp_sum += MIs[j]
            CDS_sumSI.append(temp_sum)
        totSum = 0
        for i in xrange(self.CDS_n):
            if i not in self.obsolete_id:
                totSum += CDS_sumSI[i]
        if totSum == 0:
            print CDS_sumSI, "warning"
        for i in xrange(len(CDS_sumSI)):
            CDS_sumSI[i] = CDS_sumSI[i] * 1.0 / totSum
        #
        self.CDS_SI = CDS_sumSI
    #
    def predict(self, testX):
        res = []
        for x in testX:
            y = 0
            for i in xrange(self.CDS_n):
                if i in self.obsolete_id:
                    continue
                tempx = []
                for ind, xi in enumerate(x):
                    if ind in self.MS[i]:
                        tempx.append(xi)
                if len(tempx) == 0:
                    continue
                y += self.clsfs[i].predict([tempx])[0] * self.CDS_SI[i]
            assert y >= -0.1 and y <= 1.1, y
            if y >= 0.5:
                res.append(1)
            else:
                res.append(0)
        return res

class IncomSelected:
    def __init__(self, clsfier, alpha = 0.6, weight = 'auto'):
        self.clsfier = clsfier
        self.alpha = alpha
        self.weight_kind = weight

    def fit(self, trainX, trainY):
        self.trainX = trainX
        self.trainY = trainY
        self.row_n = len(trainY)
        self.col_n = len(trainX[0])
        assert self.row_n == len(trainX)

        MvS = [] # for every data
        for itt in trainX: 
            numb = set()
            for ind, x in enumerate(itt):
                if x != None and x != '?': 
                    numb.add(ind)
            MvS.append(tuple(numb))
        # dataX[i] is a sample
        # MvS[i] is the no missing features of this sample.
        # eg: MvS[3] = (0, 1, 3) 
        MS = tuple(set(MvS)) 
        self.MS = MS  

        CDS = [] 
        for i in MS: 
            # i is some kind of MvS 
            #eg: i = (0, 1) 
            CDSi = []
            for ind, mvs in enumerate(MvS): 
                if set(i).issubset(set(mvs)): 
                    CDSi.append(ind)
            CDS.append(CDSi)
        self.CDS = CDS
        self.CDS_n = len(CDS)
        # CDS[3] = [id1, id2, id3, ...] 

        MIs = []
        for i in xrange(self.col_n): 
            x = []
            y = []
            for j in xrange(self.row_n):
                if self.trainX[j][i] != None:
                    x.append(self.trainX[j][i])
                    y.append(self.trainY[j])   
            MIs.append(hhkutil.MI(x, y))
        #
        # print MIs
        CDS_sumSI = []
        for i in xrange(self.CDS_n):
            #for every CDS
            temp_sum = 0
            for j in MS[i]:
                temp_sum += MIs[j]
            CDS_sumSI.append(temp_sum)
        
        ############################
        ## select_CDS
        alpha = self.alpha
        begin_time = time.clock()

        colX = self.col_n
        rowX = self.row_n
        U = set(range(0, colX - 1))
        lenT = rowX
        L = set()
        M = set()
        temp_CDS = set(range(len(self.MS)))
        while len(U) > 0 or len(M) < alpha * lenT:
            W = []
            if len(M) >= alpha * lenT:
                W = [1] * len(self.MS)
            else:
                for i in self.CDS:
                    W.append(len(set(i) - M))
            #select
            max_value = -1
            max_id = -1
            max_fea_set = set()
            for i in temp_CDS:
                cur_fea_set = set(self.MS[i])
                value_last = len(U & cur_fea_set) # | the feature set of CDSi & U |
                cur_value = W[i] * CDS_sumSI[i] * value_last
                if cur_value > max_value:
                    max_id = i
                    max_value = cur_value
                    max_fea_set = cur_fea_set
            #
            # if len(max_fea_set) < (colX - 1):   # for the CDS that all columns are included
                # U = U - max_fea_set
            U = U - max_fea_set
            L.add(max_id)
            temp_CDS = temp_CDS - set([max_id])
            M = M | set(self.CDS[max_id])

        print L
        print "select end", time.clock() - begin_time
        self.select_CDS_n = len(L)
        # L contains the indexes of CDS
        self.select_CDS_id = L

        ## selct end

        clsfs = dict()
        for ind in self.select_CDS_id:
            fea = self.MS[ind]
            nclsfier = copy.deepcopy(self.clsfier)
            if len(fea) == 0:
                # clsfs.append(nclsfier)
                continue
            tempX = []
            tempY = []
            for j in CDS[ind]:
                x = []
                for col in fea:
                    x.append(self.trainX[j][col])
                tempX.append(x)
                tempY.append(self.trainY[j])
            #
            # print fea
            # for x in tempX:
            #     print x
            #
            nclsfier.fit(tempX, tempY)
            clsfs[ind] = nclsfier
        self.clsfs = clsfs

        totSum = 0
        for i in self.select_CDS_id:
            totSum += CDS_sumSI[i]
        if totSum == 0:
            print CDS_sumSI, "warning"
        CDS_SI = dict()
        if self.weight_kind == 'auto':
            for i in self.select_CDS_id:
                CDS_SI[i] = CDS_sumSI[i] * 1.0 / totSum
        elif self.weight_kind == 'same':
            for i in self.select_CDS_id:
                CDS_SI[i] = 1.0 / self.select_CDS_n
        else:
            print "self.weight_kind error!"
        self.CDS_SI = CDS_SI
    #
    def predict(self, testX):
        res = []
        for x in testX:
            y = 0
            for i in self.select_CDS_id:
                tempx = []
                for ind, xi in enumerate(x):
                    if ind in self.MS[i]:
                        tempx.append(xi)
                if len(tempx) == 0:
                    continue
                y += self.clsfs[i].predict([tempx])[0] * self.CDS_SI[i]
            assert y >= -0.1 and y <= 1.1, y
            if y >= 0.5:
                res.append(1)
            else:
                res.append(0)
        return res

class IncomWeighed:
    def __init__(self, clsfier, alpha = 0.6):
        self.clsfier = clsfier
        self.alpha = alpha

    def fit(self, trainX, trainY):
        self.trainX = trainX
        self.trainY = trainY
        self.row_n = len(trainY)
        self.col_n = len(trainX[0])
        assert self.row_n == len(trainX)

        MvS = [] # for every data
        for itt in trainX: 
            numb = set()
            for ind, x in enumerate(itt):
                if x != None and x != '?': 
                    numb.add(ind)
            MvS.append(tuple(numb))
        # dataX[i] is a sample
        # MvS[i] is the no missing features of this sample.
        # eg: MvS[3] = (0, 1, 3)
        MS = tuple(set(MvS)) 
        self.MS = MS  

        CDS = []
        for i in MS: 
            # i is some kind of MvS
            #eg: i = (0, 1)
            CDSi = []
            for ind, mvs in enumerate(MvS):
                if set(i).issubset(set(mvs)): 
                    CDSi.append(ind)
            CDS.append(CDSi)
        self.CDS = CDS
        self.CDS_n = len(CDS)
        # CDS[3] = [id1, id2, id3, ...]

        MIs = []
        for i in xrange(self.col_n): 
            x = []
            y = []
            for j in xrange(self.row_n):
                if self.trainX[j][i] != None:
                    x.append(self.trainX[j][i])
                    y.append(self.trainY[j])   
            MIs.append(hhkutil.MI(x, y))
        #
        # print MIs
        CDS_sumSI = []
        for i in xrange(self.CDS_n):
            #for every CDS
            temp_sum = 0
            for j in MS[i]:
                temp_sum += MIs[j]
            CDS_sumSI.append(temp_sum)
        
        ############################
        ## select_CDS
        alpha = self.alpha
        begin_time = time.clock()

        colX = self.col_n
        rowX = self.row_n
        U = set(range(0, colX - 1))
        lenT = rowX
        L = set()
        M = set()
        temp_CDS = set(range(len(self.MS)))
        while len(U) > 0 or len(M) < alpha * lenT:
            W = []
            if len(M) >= alpha * lenT:
                W = [1] * len(self.MS)
            else:
                for i in self.CDS:
                    W.append(len(set(i) - M))
            #select
            max_value = -1
            max_id = -1
            max_fea_set = set()
            for i in temp_CDS:
                cur_fea_set = set(self.MS[i])
                value_last = len(U & cur_fea_set) # | the feature set of CDSi & U |
                cur_value = W[i] * CDS_sumSI[i] * value_last
                if cur_value > max_value:
                    max_id = i
                    max_value = cur_value
                    max_fea_set = cur_fea_set
            #
            # if len(max_fea_set) < (colX - 1):   # for the CDS that all columns are included
                # U = U - max_fea_set
            U = U - max_fea_set
            L.add(max_id)
            temp_CDS = temp_CDS - set([max_id])
            M = M | set(self.CDS[max_id])

        print L
        print "select end", time.clock() - begin_time
        print len(L)
        # L contains the indexes of CDS
        self.select_CDS_id = L
        self.select_CDS_n = len(L)
        ## selct end

        clsfs = dict()
        for ind in self.select_CDS_id:
            fea = self.MS[ind]
            nclsfier = copy.deepcopy(self.clsfier)
            if len(fea) == 0:
                print ">>>>><<<<<<"
                clsfs.append(nclsfier)
                continue
            tempX = []
            tempY = []
            for j in CDS[ind]:
                x = []
                for col in fea:
                    x.append(self.trainX[j][col])
                tempX.append(x)
                tempY.append(self.trainY[j])
            #
            # print fea
            # for x in tempX:
            #     print x
            #
            nclsfier.fit(tempX, tempY)
            clsfs[ind] = nclsfier
        self.clsfs = clsfs
    #
    def trainWeight(self, tweightX, tweightY, fx):
        lr = copy.deepcopy(fx)
        res = []
        for i in self.select_CDS_id:
            tempx = []
            for x in tweightX:
                tempxi = []
                for ind, xi in enumerate(x):
                    if ind in self.MS[i]:
                        tempxi.append(xi)
                tempx.append(tempxi)
            res.append(self.clsfs[i].predict(tempx))
        lrfx = np.array(res).T
        lr.fit(lrfx, tweightY)
        # print "lr.coef_", lr.coef_
        print type(lr)
        coef = lr.coef_
        print coef
        tot = sum(coef)
        for i in xrange(self.select_CDS_n):
            coef[i] = coef[i] * 1.0 / tot
        self.CDS_wi = dict(zip(self.select_CDS_id, coef))
        print "trainWeight end"
        print "CDS_wi:"
        for i in self.CDS_wi:
            print i, self.CDS_wi[i]
    def predict(self, testX):
        res = []
        for x in testX:
            y = 0
            for i in self.select_CDS_id:
                tempx = []
                for ind, xi in enumerate(x):
                    if ind in self.MS[i]:
                        tempx.append(xi)
                if len(tempx) == 0:
                    continue
                y += self.clsfs[i].predict([tempx])[0] * self.CDS_wi[i]
            # assert y >= -0.1 and y <= 1.1, y
            if y >= 0.5:
                res.append(1)
            else:
                res.append(0)
        return res

class IN_cmp1:
    # ANN with some kinds of Adaboost
    # the idea comes from a paper
    def __init__(self, debug = 0):
        # string
        self.debug = debug
    def fit(self, trainX, trainY, trainX2, trainY2):
        self.trainX = trainX
        self.trainY = trainY
        self.row_n = len(trainY)
        self.col_n = len(trainX[0])

        MvS = [] # for every data
        for X in trainX:
            numb = set()
            for i in xrange(self.col_n):
                if X[i] != None and X[i] != '?': 
                    numb.add(i)
            MvS.append(tuple(numb))
        # dataX[i] is a sample
        # MvS[i] is the no missing features of this sample.
        # eg: MvS[3] = (0, 1, 3)
        MS = tuple(set(MvS)) 
        self.MS = MS  
        
        CDS = []
        for i in MS:   
            # i is some kind of MvS
            #eg: i = (0, 1)
            CDSi = []
            for j in xrange(len(MvS)):
                if set(i).issubset(set(MvS[j])): 
                    CDSi.append(j)
            CDS.append(CDSi)
        self.CDS = CDS
        # CDS[3] = [id1, id2, id3, ...]

        ## train ANN Ada 
        classifier_for_every_DS = []

        for index, i in enumerate(self.CDS):
            L = len(i)**0.5
            temp_X = []
            temp_Y = []
            for i1 in i:
                #i1 is the index of some data in dataX
                cur_x = []
                for i2 in self.MS[index]:
                    cur_x.append(trainX2[i1][i2])
                temp_X.append(cur_x)
                temp_Y.append(trainY2[i1])
            cur_coln = len(temp_X[0])
            if len(temp_X) == 0 or cur_coln == 0:
                classifier_for_every_DS.append(None)
                continue
            bdt = SAda(
                MLPClassifier(tol = 0.001, max_iter = 800), 
                max_iter = int(L * 0.7)
                )
            
            bdt.fit(temp_X, temp_Y)
            classifier_for_every_DS.append(bdt)

        self.classifier_for_every_DS = classifier_for_every_DS
    #
    def predict(self, testX):
        res = []
        for x in testX:
            sump1 = 0
            sump2 = 0
            for index1, classifier in enumerate(self.classifier_for_every_DS):
                if type(classifier) == type(None):
                    continue
                temp_X = []
                for i2 in self.MS[index1]:
                    temp_X.append(x[i2])
                p1, p2 = classifier.predict_proba([temp_X])[0]
                
                sump1 += p1 
                sump2 += p2 
            if sump1 > sump2:
                res.append(0)
            else:
                res.append(1)
        return res
#
class SAda:
    # till now, just for the class above
    def __init__(self, clf, max_iter):
        self.clf = clf
        self.max_iter = max_iter
    def fit(self, trainX, trainY):
        classifier = []
        classifier_wi = []
        inputx = copy.deepcopy(trainX)
        inputy = copy.deepcopy(trainY)
        self.trainX = list(trainX)
        self.trainY = list(trainY)
        self.col_n = len(self.trainX[0])
        last_e = 0
        for n in xrange(self.max_iter):
            mlp = copy.deepcopy(self.clf)
            mlp.fit(self.trainX, self.trainY)
            res = mlp.predict(inputx)
            cur_row_n = len(self.trainX)
            right_n = 0
            false_n = 0
            for ind, y in enumerate(res):
                if y != inputy[ind]:
                    false_n += 1
                    self.trainX.append(inputx[ind])
                    self.trainY.append(inputy[ind])
                else:
                    right_n += 1
            cur_e = right_n * 1.0 / (right_n + false_n)
            if cur_e < last_e:
                break
            last_e = cur_e
            if cur_e > 0.6:
                classifier.append(mlp)
                classifier_wi.append(cur_e * len(inputx) * self.col_n)

        self.classifier = classifier
        self.classifier_wi = classifier_wi
    def predict_proba(self, testX):
        res = []
        for x in testX:
            pro1, pro2 = 0, 0
            for ind, clfs in enumerate(self.classifier):
                y1, y2 = clfs.predict_proba([x])[0]
                pro1 += y1 * self.classifier_wi[ind]
                pro2 += y2 * self.classifier_wi[ind]
                # if clfs.predict([x])[0] == 0:
                #     pro1 += self.classifier_wi[ind]
                # else:
                #     pro2 += self.classifier_wi[ind]
            res.append((pro1, pro2))
        return res

class IN_cmp2:
    # make a tree to store many kinds of missing condition
    # the idea comes from another paper
    def __init__(self, clf):
        # string
        self.debug = False
        self.clf = clf
    def fit(self, trainX, trainY):
        self.trainX = trainX
        self.trainY = trainY
        self.row_n = len(trainX)
        self.col_n = len(trainX[0]) 
        
        CDS_tree = dict()
        
        MS = []
        # MS[i] = (1, 2) 
        # the i-th sample in dataX miss the (1, 2)
        for ind, X in enumerate(trainX):
            numb = set()
            for i in xrange(self.col_n):
                if X[i] == None or X[i] == '?': 
                    numb.add(i)
            temp_miss = tuple(numb)
            MS.append(temp_miss)
        # 
        classifier_key = set(MS)
        for i in classifier_key:
            CDS_tree[i] = set()
        for index, i in enumerate(MS):
            for j in classifier_key:
                if set(i).issubset(set(j)):
                    CDS_tree[j].add(index)
        #
        self.CDS_tree = CDS_tree
        self.MS = MS

        #  make MI 
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
        #
        ## train classfier
        tree_classifier = dict()
        wi = dict()
        print len(self.CDS_tree)
        for i in self.CDS_tree:
            t_trainX = []
            t_trainY = []
            for ind in self.CDS_tree[i]:
                curX = []
                for j in xrange(self.col_n):
                    if j not in i:
                        curX.append(self.trainX[ind][j])
                t_trainX.append(curX)
                t_trainY.append(self.trainY[ind])
            if len(t_trainX) == 0 or len(t_trainX[0]) == 0:
                # all missing
                continue

            mlp = copy.deepcopy(self.clf)
            mlp.fit(t_trainX, t_trainY)
            tree_classifier[i] = mlp
            
            cur_wi = 0
            for j in xrange(self.col_n):
                if j not in i:
                    cur_wi += self.MIs[j]
            cur_wi *= len(self.CDS_tree[i])
            
            right_n = 0
            false_n = 0
            ansY = mlp.predict(t_trainX)
            for ind2, j in enumerate(ansY):
                if j == trainY[ind2]:
                    right_n += 1
                else:
                    false_n += 1
            acc = right_n * 1.0 / (right_n + false_n)
            # if self.debug == True:
                # print "acc", acc
            cur_wi *= acc
            wi[i] = cur_wi
        #
        self.tree_classifier = tree_classifier
        self.wi = wi

    #
    def predict(self, testX):
        ansY = []
        for curX in testX:
            numb = set()
            for ind, x in enumerate(curX):
                if x == None or x == '?': 
                    numb.add(ind)
            miss_f = tuple(numb)

            if len(miss_f) == self.col_n:
                ansY.append(-2)
                continue
            
            classifiers_miss = [miss_f]
            for col in xrange(self.col_n):
                if col not in miss_f:
                    tp_missf = tuple(set(miss_f + (col,)))
                    if self.tree_classifier.has_key(tp_missf):
                        classifiers_miss.append(tp_missf)
            #
            y = 0
            for miss in classifiers_miss:
                if self.tree_classifier.has_key(miss) == False:
                    continue
                x = []
                for col in xrange(self.col_n):
                    if col not in miss:
                        x.append(curX[col])
                y += self.tree_classifier[miss].predict([x])[0] * self.wi[miss]
            if y >= 0.5:
                ansY.append(1)
            else:
                ansY.append(0)
        return ansY
    #   
#
############  cmp3
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


if __name__ == "__main__":
    print "main: incomp.py"