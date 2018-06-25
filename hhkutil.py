import collections
import math

def MI(x, y):
    dict_x = dict(collections.Counter(x))
    len_x = len(x)
    dict_y = dict(collections.Counter(y))
    len_y = len(y)
    x_y = []
    for i in xrange(len_x):
        x_y.append((x[i], y[i]))
    dict_x_y = dict(collections.Counter(x_y))
    ans = 0.0
    for yy in dict_y:
        for xx in dict_x:
            p_x = dict_x[xx] * 1.0 / len_x
            p_y = dict_y[yy] * 1.0 / len_y
            if dict_x_y.has_key((xx, yy)):
                p_x_y = dict_x_y[(xx, yy)] 
                ans += p_x_y * math.log(1.0 * p_x_y / p_x / p_y)
    if (ans < 0):
        print "warning at MI"
    return ans
#
def calres(res_Y, lable, debug = 0):
    anyone = -1
    jumpone = -2
    print "calc_pre_and_rec begin: len res_Y:", len(res_Y), "len lable", len(lable)
    #print res_Y
    #print lable
    if debug >= 2:
        for ind, x in enumerate(res_Y):
            print x, lable[ind]
    if len(res_Y) != len(lable):
        print "error lenth"
        return
    if type(res_Y[0]) != type(lable[0]):
        print "warning type", type(res_Y[0]), type(lable[0])
    i = 0
    sta = dict()       # TP, FP, FN 
    while i < len(res_Y):
        if res_Y[i] == jumpone:
            i += 1
            continue
        if res_Y[i] == lable[i]:
            if sta.has_key(lable[i]):
                sta[lable[i]][0] += 1
                sta[lable[i]][3] += 1
            else :
                sta[lable[i]] = [1, 0, 0, 1]
        else :
            if sta.has_key(lable[i]):
                sta[lable[i]][2] += 1
                sta[lable[i]][3] += 1
            else :
                sta[lable[i]] = [0, 0, 1, 1]
            if sta.has_key(res_Y[i]):
                sta[res_Y[i]][1] += 1
            else :
                sta[res_Y[i]] = [0, 1, 0, 0]
        i += 1
    #
    if debug >= 1:
        for i in sta:
            if sta[i][0] == 0:
                print "###", i, 0, 0
                continue
            print "###", i, sta[i][0] * 1.0 / (sta[i][0] + sta[i][1]), sta[i][0] * 1.0 / (sta[i][0] + sta[i][2])
        # precesion = TP / (TP + FP),  recall = TP / (TP + FN)
    #
    # calc average precesion and recall
    pre = 0
    rec = 0
    cnt = 0
    for i in sta:
        cnt += sta[i][3]
        if sta[i][0] == 0:
            continue
        pre += sta[i][0] * 1.0 * sta[i][3] / (sta[i][0] + sta[i][1])
        rec += sta[i][0] * 1.0 * sta[i][3] / (sta[i][0] + sta[i][2])
    if cnt != len(res_Y):
        print "warning:", cnt, len(res_Y)
    if cnt == 0:
        return -1, -1
    r_pre = pre / cnt
    r_rec = rec / cnt
    return r_pre, r_rec
#
def subSet(x):
    # return a set of tuple
    assert type(x) == type([]) or type(x) == type(tuple())
    num = len(x)
    ans = set()
    for i in xrange(1,1<<num):
        f = i
        idx = 0
        tp = set()
        while f > 0:
            if f % 2 == 1:
                tp.add(x[idx])
            f /= 2
            idx += 1
        ans.add(tuple(tp))
    return ans
#
def normaled(x, sum_x = 1.0, debug = 0):
    if type(x) == type(dict()):
        tot = 0
        for i in x:
            tot += x[i]
        if tot <= 0.001 and tot >= -0.001:
            print "normaled warning"
            for i in x:
                print i, x[i]
        assert tot > 0.001 or tot < -0.001, tot
        y = dict()
        for i in x:
            y[i] = x[i] * 1.0 / tot
        return y
    assert type(x) == type([]) or type(x) == type(tuple())
    tot = 0
    for i in x:
        if i < 0:
            # print "warning: i<0:", i
            pass
        tot += i
    # print "x:\n", x
    # print "tot =", tot
    if tot <= 0.0001 and tot >= -0.0001:
        # print "warning"
        if debug == 1:
            return x, 1
        return x 
    res = [i * sum_x * 1.0 / tot for i in x]
    if debug == 1:
        return res, 0
    return res