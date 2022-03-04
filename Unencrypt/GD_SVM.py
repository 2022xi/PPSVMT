
# from matplotlib import  pyplot as plt

from modelTrain import *
from experiment_method import *





BCW = "01breast-cancer-wisconsin.txt"
WDBC = "02wdbc.txt"
WPBC = "03wpbc.txt"
PID = "04Pima Indian Diabetic.txt"
HDD = "05Heart Disease Data Set.txt"  # "heart disease data.txt"

T = 10
"***数据标准化方法选择***"
# stand_method = normalization1
# stand_method = linear2
# stand_method = range3
stand_method = standard4

"***评估方法选择***"
# evaluation_method = holdout_train
evaluation_method = cross_train
# evaluation_method = one_train

"***模型训练方法选择***"
# train_method = old_model
# train_method = exp_model
# train_method = log_model
# generate a public key and private key pair
model_name = ["exp_model", "log_model"]
aa = 0
for train_method in [old_model]:
    with open("result/"+"result.txt", 'w') as f:
        aa += 1
        for data_set in [BCW,WDBC,WPBC,PID,HDD]:
            jdata, ndata, jtabel, ntabel = read_data(data_set, stand_method)

            TPsum, FPsum, FNsum, TNsum, iter_num, tsum, run_time, DPtime_sum, DACtime_sum = evaluation_method(jdata, ndata, jtabel, ntabel, T, train_method)
            print(data_set,file=f)
            print("平均训练耗时：%fs， DP耗时：%fs， DAC耗时：%fs" % (run_time/10, DPtime_sum/10, DACtime_sum/10),file=f)
            total = TPsum+FPsum+FNsum+TNsum
            print("%d次测试迭代总数：%d，平均迭代次数：%d" % (10, iter_num, iter_num/10),file=f)
            print("%d次测试样本总数：%d，分类正确总数：%d，正确率：%.3f" % (10, total, tsum, tsum/total),file=f)
            print("%d次总和 TP=%d,FP=%d,FN=%d,TN=%d\n" % (10, TPsum, FPsum, FNsum, TNsum),file=f)

