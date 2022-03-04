# PPSVMT
This project aims at the status quo of "information silos" and accomplishes federal learning under privacy protection. Based on Paillier encryption system, model parameters are encrypted, gradients are updated locally and SVM model training is completed jointly, which can protect data samples and model training parameters from leakage at the same time. Paillier encryption system has the characteristics of addition and number multiplication homomorphism, but the ordinary SVM model training algorithm is complex, not only contains addition and number multiplication two operations, so we take exponential second-order Taylor expansion as the loss function, using gradient descent method to locally optimize the model parameters. Through federated learning, the high-precision SVM model is finally trained under the condition of protecting data samples and model training parameters.
