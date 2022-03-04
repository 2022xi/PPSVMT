import numpy as np
import random
import math


def labelStand(y, data_set):
    if data_set == 1:
        if y == 2:
            return -1   # 标签为 0,1 代表不存在心脏病，返回-1
        else:
            return 1    # 标签为 2,3,4 代表存在心脏病，返回1
    elif data_set == 0:
        if y < 2:
            return -1  # 标签为 0,1 代表不存在心脏病，返回-1
        else:
            return 1  # 标签为 2,3,4 代表存在心脏病，返回1



def normalization1(jdata, ndata, jnum, nnum):
    col_sum2 = []
    n = jdata.shape[1]
    for j in range(n):
        asum2 = 0
        for i in range(jnum):
            # print(jdata[i, j],jdata[i, j]>0)
            asum2 += int(jdata[i, j])^2

        for i in range(nnum):
            asum2 += int(ndata[i, j])^2

        col_sum2.append(math.sqrt(asum2))
    col_sum2 = np.mat(col_sum2)
    print("向量根号平方和：", col_sum2)
    jdata = jdata / col_sum2
    ndata = ndata / col_sum2
    return jdata, ndata


def linear2(jdata, ndata, jnum, nnum):
    col_max = []
    n = jdata.shape[1]
    for j in range(n):
        amax = 0
        for i in range(jnum):
            if jdata[i, j] > amax:
                amax = jdata[i, j]

        for i in range(nnum):
            if ndata[i, j] > amax:
                amax = ndata[i, j]
        col_max.append(amax)
    col_max = np.mat(col_max)
    print("各属性中的最大值：", col_max)
    jdata = jdata / col_max
    ndata = ndata / col_max
    return jdata, ndata

def range3(jdata, ndata, jnum, nnum):
    col_max = []
    col_min = []
    n = jdata.shape[1]
    for j in range(n):
        amax = 0
        amin = 10000
        for i in range(jnum):
            if jdata[i, j] > amax:
                amax = jdata[i, j]
            if jdata[i, j] < amin:
                amin = jdata[i, j]

        for i in range(nnum):
            if ndata[i, j] > amax:
                amax = ndata[i, j]
            if ndata[i, j] < amin:
                amin = ndata[i, j]

        col_max.append(amax)
        col_min.append(amin)
    col_max = np.mat(col_max)
    col_min = np.mat(col_min)
    print("各属性中的最大值：", col_max)
    print("各属性中的最小值：", col_min)
    jdata = (jdata - col_min) / (col_max - col_min)
    ndata = (ndata - col_min) / (col_max - col_min)

    return jdata, ndata

def standard4(jdata, ndata, jnum, nnum):
    num = jnum + nnum
    avex = (np.sum(jdata, axis=0) + np.sum(ndata, axis=0)) / num
    jdiff = jdata - avex
    ndiff = ndata - avex
    jdiff2 = np.multiply(jdiff, jdiff)
    ndiff2 = np.multiply(ndiff, ndiff)
    dsum2 = np.sum(jdiff2, axis=0) + np.sum(ndiff2, axis=0)
    S = np.sqrt(dsum2/(num-1))
    print("样本属性均值：", avex)
    print("样本属性方差：", S)
    jdata = (jdata - avex) / S
    ndata = (ndata - avex) / S
    return jdata, ndata

def read_data(data_set, stand_method):
    jdata = []
    jtabel = []
    ndata = []
    ntabel = []
    jnum = 0
    nnum = 0
    with open('dataSet/'+data_set) as file:
        for line in file:
            adata = line.strip().split(',')
            judge = 1
            for i in adata:
                if i =='?':
                    judge = 0
            if judge == 1:
                adata = list(map(float, adata))
                if adata[-1] > 0:
                    jdata.append(adata[:-1])
                    jtabel.append(1)
                    jnum += 1
                else:
                    ndata.append(adata[:-1])
                    ntabel.append(-1)
                    nnum += 1
        num = jnum + nnum
        print(data_set)
        print("样本总数：%d,正例个数：%d,反例个数：%d" % (num, jnum, nnum))
    jdata = np.mat(jdata)
    ndata = np.mat(ndata)
    jdata, ndata = stand_method(jdata, ndata, jnum, nnum)

    return jdata, ndata, jtabel, ntabel

def hold_out(jdata, ndata, jtabel, ntabel):
    jnum = len(jdata)
    nnum = len(ndata)
    train_j = jnum * 7 // 10
    train_n = nnum * 7 // 10
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    # 正例中随机划分
    subscript = list(range(jnum))
    random.shuffle(subscript)
    j = 1
    for i in subscript:
        if j <= train_j:
            data_stand = jdata[i]
            train_x.append(data_stand.T)
            train_y.append(jtabel[i])
            j +=1
        else:
            data_stand = jdata[i]
            test_x.append(data_stand.T)
            test_y.append(jtabel[i])

    # 反例中随机划分
    subscript = list(range(nnum))
    random.shuffle(subscript)
    j = 1
    for i in subscript:
        if j <= train_n:
            data_stand = ndata[i]
            train_x.append(data_stand.T)
            train_y.append(ntabel[i])
            j +=1
        else:
            data_stand = ndata[i]
            test_x.append(data_stand.T)
            test_y.append(ntabel[i])

    train_num = train_n+train_j
    test_num = jnum+nnum - train_n-train_j
    sum_1 = 0
    for i in range(test_num):
        if test_y[i] ==1 :
            sum_1 +=1
    print("样本总数：%d,训练集数量：%d,测试集数量：%d" % (jnum+nnum, train_num, test_num))
    print("测试集总数：%d,正例个数：%d,反例个数：%d" %(test_num, sum_1, test_num-sum_1))
    return train_x, train_y, test_x, test_y, train_num, test_num

def cross_validation(jdata, ndata, jtabel, ntabel):
    jnum = len(jdata)
    nnum = len(ndata)
    data_xi = []
    data_yi = []
    for _ in range(10):
        data_xi.append([])
        data_yi.append([])

    # 正例中随机划分
    subscript = list(range(jnum))
    random.shuffle(subscript)
    j = 0
    for i in subscript:
        data_stand = jdata[i]
        data_xi[j].append(data_stand.T)
        data_yi[j].append(jtabel[i])
        if j < 9:
            j +=1
        else:
            j = 0

    # 反例中随机划分
    subscript = list(range(nnum))
    random.shuffle(subscript)
    j = 0
    for i in subscript:
        data_stand = ndata[i]
        data_xi[j].append(data_stand.T)
        data_yi[j].append(ntabel[i])
        if j < 9:
            j +=1
        else:
            j = 0

    astr = ''
    for a in data_yi:
        astr += '%d ' % len(a)
    print("样本总数：%d,集合样本数量：%s" % (jnum + nnum, astr))

    return data_xi, data_yi