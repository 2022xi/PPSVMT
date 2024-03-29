# PPSVMT
## 介绍

原创论文“Privacy-Preserving Online Medical Prediagnosis Training Model Based on Soft-Margin SVM”

该项目针对“信息孤岛”现状，基于Paillier密码系统对模型参数进行加密，数据提供商计算局部梯度，模型训练方整合梯度并更新模型参数，同时保护各方数据样本和模型参数不被泄漏，共同完成SVM模型的训练。Paillier加密系统具有加法和数乘同态的特性，但是寻常的SVM模型训练算法计算复杂，于是我们以指数二阶泰勒展开式作为损失函数，使训练过程中只包含加法和数乘运算，采用梯度下降法对模型参数进行优化，通过联邦学习最终训练出高精度的SVM模型。
