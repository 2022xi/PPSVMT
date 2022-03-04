from modelTest import *
from data_divide import *
import time



def cross_train(jdata, ndata, jtabel, ntabel, T, train_method):  # 交叉验证法测试
    iter_num = 0  # 迭代次数总和
    TPsum = 0  # 预测正例中正例的个数
    FNsum = 0  # 预测正例中反例的个数
    FPsum = 0  # 预测正例中正例的个数
    TNsum = 0  # 预测正例中反例的个数
    tsum = 0  # 预测正确的总数
    DPtime_sum = 0
    DACtime_sum = 0

    # 交叉验证法
    T = T//10
    run_time = 0
    for j in range(T):
        data_xi, data_yi = cross_validation(jdata, ndata, jtabel, ntabel)
        # print(len(data_yi))
        for i in range(10):
            print("\n第%d次训练测试" % (j*10 + i + 1))
            test_x = data_xi[i]
            test_y = data_yi[i]

            test_num = len(test_y)
            sum_1 = 0
            for k in test_y:
                if k > 0:
                    sum_1 += 1

            train_x = []
            train_y = []
            for k in range(10):
                if k != i:
                    train_x += data_xi[k]
                    train_y += data_yi[k]
            train_num = len(train_y)

            print("样本总数：%d,训练集数量：%d,测试集数量：%d" % (len(jdata)+len(ndata), train_num, test_num))
            print("测试集总数：%d,正例个数：%d,反例个数：%d" %(test_num, sum_1, test_num-sum_1))
            begin_time = time.time()
            omega, b, t = train_method(train_x, train_y, train_num)
            end_time = time.time()
            run_time += end_time - begin_time

            iter_num += t
            # Ex = 0  # 正例的个数
            # Count = 0  # 反例的个数
            test_num = len(test_y)
            asum, TP, FN, FP, TN = modelTest(omega, b, test_x, test_y, test_num)
            tsum += asum
            TPsum += TP
            FPsum += FP
            FNsum += FN
            TNsum += TN
            print("该次测试样本总数：%d，分类正确总数：%d，正确率：%.3f" % (test_num, asum, asum / test_num))
            print("TP=%d,FP=%d,FN=%d,TN=%d" % (TP, FP, FN, TN))
    return TPsum, FPsum, FNsum, TNsum, iter_num, tsum, run_time, DPtime_sum, DACtime_sum

