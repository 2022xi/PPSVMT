import numpy as np
import math


def old_model(train_x, train_y, m):
    pcostlist = []
    costlist = []
    omega = np.zeros(train_x[0].shape)
    b = 0
    print("omega、b初值：",omega.T,b)
    lam = 0.01
    T = 500
    precision = 1e-3
    cost = 10
    C = 1/m
    print("C值：",C,",lam值：",lam)
    t = 0
    fcost = cost
    pcost = 1
    I = [0]*m

    while t < T: #pcost > precision and
        delta = omega.copy()
        delta_b = 0
        for i in range(m):
            a = train_y[i] * (omega.T * train_x[i] + b) # a = train_y[i] * (omega.T * train_x[i] + b)
            if a < 1:
                delta -= C*train_y[i]*train_x[i]
                delta_b -= C*train_y[i]
                I[i] = 1

        omega -= lam*delta
        b -= lam*delta_b
        omega_norm = np.linalg.norm(omega, 2)
        cost = (omega_norm**2)/2
        for i in range(m):
            if I[i]:
                temp = 1 - train_y[i] * (omega.T * train_x[i] + b)
                cost += C*temp[0,0]

        # print(cost)
        pcost = abs(cost - fcost)
        # print(pcost)
        fcost = cost
        pcostlist.append(pcost)
        costlist.append(cost)
        if t == 1:
            pcost +=1
        t += 1
    print("迭代次数：",t)
    return omega, b, t #, pcostlist, costlist

def exp_model(train_x, train_y, m):
    omega = np.zeros(train_x[0].shape)
    b = 0
    print("omega、b初值：",omega.T,b)
    lam = 0.01
    T = 500
    precision = 1e-3
    cost = 10
    C = 1/m
    print("C值：",C,",lam值：",lam)
    t = 0
    fdelta = np.zeros(train_x[0].shape)
    fdelta_b = 0
    while t < T: #pcost > precision and
        delta = omega.copy()
        delta_b = 0
        for i in range(m):
            a = train_y[i] * (omega.T * train_x[i] + b) -1 # a = train_y[i] * (omega.T * train_x[i] + b)
            delta += C*train_y[i]*train_x[i]*a
            delta_b += C*train_y[i]*a

        omega -= lam*delta
        b -= lam*delta_b
        cost = np.linalg.norm(np.append(delta - fdelta,delta_b - fdelta_b), 2)
        fdelta = delta
        fdelta_b = delta_b
        t += 1
    print("迭代次数：",t)
    return omega, b, t

def log_model(train_x, train_y, m):
    omega = np.zeros(train_x[0].shape)
    b = 0
    print("omega、b初值：",omega.T,b)
    lam = 0.01
    T = 500
    precision = 1e-3
    cost = 10
    C = 1/m
    print("C值：",C,",lam值：",lam)
    t = 0
    fdelta = np.zeros(train_x[0].shape)
    fdelta_b = 0
    while t < T: #cost > precision and
        delta = omega.copy()
        delta_b = 0
        for i in range(m):
            a = train_y[i] * (omega.T * train_x[i] + b) # a = train_y[i] * (omega.T * train_x[i] + b)
            expa = math.exp(-a[0,0])
            delta += C*train_y[i]*train_x[i]*expa/(1+expa)
            delta_b += C*train_y[i]*expa/(1+expa)

        omega -= lam*delta
        b -= lam*delta_b
        cost = np.linalg.norm(np.append(delta - fdelta,delta_b - fdelta_b), 2)
        fdelta = delta
        fdelta_b = delta_b
        t += 1
    print("迭代次数：",t)
    return -omega, -b, t


def log_model1(train_x, train_y, m):
    ''' 对数函数二阶展开 '''
    omega = np.zeros(train_x[0].shape)
    b = 0
    print("omega、b初值：",omega.T,b)
    lam = 0.01
    T = 500
    precision = 1e-3
    cost = 10
    C = 1/m
    print("C值：",C,",lam值：",lam)
    t = 0
    fdelta = np.zeros(train_x[0].shape)
    fdelta_b = 0
    while t < T: #pcost > precision and
        delta = omega.copy()
        delta_b = 0
        for i in range(m):
            a = train_y[i] * (omega.T * train_x[i] + b) +2 # a = train_y[i] * (omega.T * train_x[i] + b)
            delta += C*train_y[i]*train_x[i]*a/4
            delta_b += C*train_y[i]*a/4

        omega -= lam*delta
        b -= lam*delta_b
        cost = np.linalg.norm(np.append(delta - fdelta,delta_b - fdelta_b), 2)
        fdelta = delta
        fdelta_b = delta_b
        t += 1
    print("迭代次数：",t)
    return -omega, -b, t
