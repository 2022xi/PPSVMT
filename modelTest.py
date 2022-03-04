def modelTest(omega, b, test_x, test_y, test_num):
    asum = 0
    TP = 0  # 预测正例中正例的个数
    FN = 0  # 预测正例中反例的个数
    FP = 0  # 预测正例中正例的个数
    TN = 0  # 预测正例中反例的个数

    for i in range(test_num):
        trainlable = (omega.T * test_x[i] + b)[0,0]
        # lablelist.append(trainlable)  # 记录运算结果，检验分类的正确性
        if trainlable > 0:
            if test_y[i] > 0:
                asum += 1
                TP += 1
            else:
                FP += 1
        elif trainlable < 0:
            if test_y[i] < 0:
                asum += 1
                TN += 1
            else:
                FN += 1
    return asum, TP, FN, FP, TN