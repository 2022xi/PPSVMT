
# from matplotlib import  pyplot as plt

from modelTrain import *
from experiment_method import *
from phe import paillier


dataSetList = ["01breast-cancer-wisconsin.txt","02wdbc.txt","03wpbc.txt","04Pima Indian Diabetic.txt","05Heart Disease Data Set.txt"]
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
train_method = exp_model
# train_method = old_model
# data_set = HDD
# 密钥长度
n_l = 1024

for data_set in dataSetList:
    public_key, private_key = paillier.generate_paillier_keypair(n_length=n_l)
    # generate a public key and private key pair
    # for data_set in [HDD, BCW]:
    jdata, ndata, jtabel, ntabel = read_data(data_set, stand_method)

    TPsum, FPsum, FNsum, TNsum, iter_num, tsum, run_time, DPtime_sum, DACtime_sum = evaluation_method(jdata, ndata, jtabel, ntabel, T, train_method, public_key, private_key)
    with open("results/"+data_set,'w') as f:
        print("\n平均训练耗时：%fs， DP耗时：%fs， DAC耗时：%fs" % (run_time/2, DPtime_sum/2, DACtime_sum/2), file=f)

        total = TPsum+FPsum+FNsum+TNsum
        print("%d次测试迭代总数：%d，平均迭代次数：%d" % (2, iter_num, iter_num / 2), file=f)
        try:
            print("%d次测试样本总数：%d，分类正确总数：%d，正确率：%.3f" % (2, total, tsum, tsum / total), file=f)
        except:
            print("参数失控，超出范围", file=f)
        print("%d次总和 TP=%d,FP=%d,FN=%d,TN=%d" % (2, TPsum, FPsum, FNsum, TNsum), file=f)
        try:
            print("Precision=%f,Recall=%f" % (TPsum / (TPsum + FPsum), TPsum / (TPsum + FNsum)), file=f)
        except:
            print("division by zero", file=f)


