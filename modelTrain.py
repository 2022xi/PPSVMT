import numpy as np
import time

def secure_comparison(encrypted_a, public_key, private_key):
    # DP 计时
    DPbegin = time.time()
    r1 = 5
    r2 = 1
    r3 = 3
    ear1_r2 = encrypted_a * r1 + r2
    er1_r3 = public_key.encrypt(r1+r3)
    DPend = time.time()
    DPtemp = DPend - DPbegin

    # DAC 计时
    DACbegin = time.time()
    ar1_r2 = private_key.decrypt(ear1_r2)
    r1_r3 = private_key.decrypt(er1_r3)
    DACend = time.time()
    DACtemp = DACend - DACbegin
    return ar1_r2 < r1_r3, DPtemp, DACtemp

def old_model(train_x, train_y, m, public_key, private_key):
    # secureSVM
    DPtime = 0
    DACtime = 0

    # DAC 计时
    DACbegin = time.time()
    omega = np.mat([0] * (train_x[0].shape[0])).T
    b = 0
    encrypted_omega = np.mat([public_key.encrypt(float(x[0, 0])) for x in omega]).T
    encrypted_b = public_key.encrypt(b)
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
    DACend = time.time()
    DACtime += DACend-DACbegin


    I = []
    for i in range(m):
        # DP 计时
        DPbegin = time.time()
        encrypted_a = train_y[i] * (encrypted_omega.T * train_x[i] + encrypted_b)
        DPend = time.time()
        DPtime += DPend - DPbegin

        symbol, DPtemp, DACtemp = secure_comparison(encrypted_a[0, 0], public_key, private_key)
        if symbol:
            I.append(i)
        DPtime += DPtemp
        DACtime += DACtemp

    while pcost > precision and t < T:
        # DP 计时
        DPbegin = time.time()
        encrypted_delta = encrypted_omega
        delta_b = 0

        for i in I:
            encrypted_delta = encrypted_delta - C*train_y[i]*train_x[i]
            delta_b = delta_b + train_y[i]
        encrypted_delta_b = public_key.encrypt(delta_b)
        DPend = time.time()
        DPtime += DPend - DPbegin

        # DAC 计时
        DACbegin = time.time()
        delta = np.mat([private_key.decrypt(x[0, 0]) for x in encrypted_delta]).T
        delta_b = private_key.decrypt(encrypted_delta_b)
        delta_b = -C*delta_b

        omega = omega - lam*delta
        b = b - lam*delta_b
        omega_norm = np.linalg.norm(omega, 2)
        cost = (omega_norm**2)/2
        encrypted_omega = np.mat([public_key.encrypt(float(x[0, 0])) for x in omega]).T
        encrypted_b = public_key.encrypt(b)
        encrypted_cost_tail = 0
        DACend = time.time()
        DACtime += DACend - DACbegin

        I = []
        for i in range(m):
            # DP 计时
            DPbegin = time.time()
            encrypted_a = train_y[i] * (encrypted_omega.T * train_x[i] + encrypted_b)
            DPend = time.time()
            DPtime += DPend - DPbegin

            symbol, DPtemp, DACtemp = secure_comparison(encrypted_a[0, 0], public_key, private_key)
            if symbol:
                # DP 计时
                DPbegin = time.time()
                encrypted_cost_tail = encrypted_cost_tail + (1-encrypted_a)
                I.append(i)
                DPend = time.time()
                DPtime += DPend - DPbegin
            DPtime += DPtemp
            DACtime += DACtemp

        # DAC 计时
        DACbegin = time.time()
        cost_tail = private_key.decrypt(C*encrypted_cost_tail[0, 0])
        cost += cost_tail

        pcost = abs(cost - fcost)
        fcost = cost

        if t == 1:
            pcost +=1
        t += 1
        DACend = time.time()
        DACtime += DACend - DACbegin
    print("迭代次数：",t)
    return omega, b, t, DPtime, DACtime

def exp_model(train_x, train_y, m, public_key, private_key):
    # PPSVMT
    DPtime = 0
    DACtime = 0

    # DAC 计时
    DACbegin = time.time()
    omega = np.mat([0]*(train_x[0].shape[0])).T
    b = 0
    temp_delta = np.mat([public_key.encrypt(float(x[0, 0])) for x in omega]).T
    temp_delta_b = public_key.encrypt(b)
    print("omega、b初值：",omega.T,b)
    lam = 0.01
    T = 100
    precision = 1e-3
    cost =10
    C = 1/m
    print("C值：",C,",lam值：",lam)
    t = 0
    fcost = cost
    pcost = 1
    DACend = time.time()
    DACtime += DACend-DACbegin

    # DP 计时
    DPbegin = time.time()
    sumyx = omega.copy()
    sumy = 0
    for i in range(m):
        sumyx = sumyx - train_y[i] * train_x[i]
        sumy -= train_y[i]
    DPend = time.time()
    DPtime += DPend-DPbegin
    while pcost > precision and t < T:
        # DAC 计时
        DACbegin = time.time()
        encrypted_omega = np.mat([public_key.encrypt(float(x[0,0])) for x in omega]).T
        encrypted_b = public_key.encrypt(b)
        DACend = time.time()
        DACtime += DACend - DACbegin

        # DP 计时
        DPbegin = time.time()
        encrypted_delta = temp_delta.copy()
        encrypted_delta_b = temp_delta_b
        for i in range(m):
            encrypted_a = (encrypted_omega.T * train_x[i] + encrypted_b)[0,0]
            for j in range(train_x[i].shape[0]):
                encrypted_delta[j,0] += train_x[i][j,0] * encrypted_a
            encrypted_delta_b += encrypted_a
        DPend = time.time()
        DPtime += DPend - DPbegin

        # DAC 计时
        DACbegin = time.time()
        delta = np.mat([private_key.decrypt(x[0,0]) for x in encrypted_delta]).T
        delta = omega + C * (delta + sumyx)
        delta_b = private_key.decrypt(encrypted_delta_b)
        delta_b = C*(delta_b + sumy)

        omega = omega - lam*delta
        b -= lam*delta_b
        cost = np.linalg.norm(np.vstack((omega, b)), 2)
        pcost = abs(cost - fcost)
        fcost = cost
        t += 1
        DACend = time.time()
        DACtime += DACend - DACbegin

    print("迭代次数：",t)
    return omega, b, t, DPtime, DACtime

def log_model(train_x, train_y, m, public_key, private_key):
    # 以log函数作为损失函数
    DPtime = 0
    DACtime = 0

    # DAC 计时
    DACbegin = time.time()
    omega = np.mat([0]*(train_x[0].shape[0])).T
    b = 0
    temp_delta = np.mat([public_key.encrypt(float(x[0, 0])) for x in omega]).T
    temp_delta_b = public_key.encrypt(b)
    print("omega、b初值：",omega.T,b)
    lam = 0.01
    T = 500
    precision = 1e-3
    cost =10
    C = 1/m
    print("C值：",C,",lam值：",lam)
    t = 0
    fcost = cost
    pcost = 1
    DACend = time.time()
    DACtime += DACend-DACbegin

    # DP 计时
    DPbegin = time.time()
    sumyx = omega.copy()
    sumy = 0
    for i in range(m):
        sumyx = sumyx - train_y[i] * train_x[i]
        sumy -= train_y[i]
    DPend = time.time()
    DPtime += DPend-DPbegin
    while pcost > precision and t < T:
        # DAC 计时
        DACbegin = time.time()
        encrypted_omega = np.mat([public_key.encrypt(float(x[0,0])) for x in omega]).T
        encrypted_b = public_key.encrypt(b)
        DACend = time.time()
        DACtime += DACend - DACbegin

        # DP 计时
        DPbegin = time.time()
        encrypted_delta = temp_delta.copy()
        encrypted_delta_b = temp_delta_b
        for i in range(m):
            encrypted_a = (encrypted_omega.T * train_x[i] + encrypted_b)[0,0]
            for j in range(train_x[i].shape[0]):
                encrypted_delta[j,0] += train_x[i][j,0] * encrypted_a
            encrypted_delta_b += encrypted_a
        DPend = time.time()
        DPtime += DPend - DPbegin

        # DAC 计时
        DACbegin = time.time()
        delta = np.mat([private_key.decrypt(x[0,0]) for x in encrypted_delta]).T
        delta = omega + C * (delta + sumyx)
        delta_b = private_key.decrypt(encrypted_delta_b)
        delta_b = C*(delta_b + sumy)

        omega = omega - lam*delta
        b -= lam*delta_b
        cost = np.linalg.norm(np.vstack((omega, b)), 2)
        pcost = abs(cost - fcost)
        fcost = cost
        t += 1
        DACend = time.time()
        DACtime += DACend - DACbegin

    print("迭代次数：",t)
    return omega, b, t, DPtime, DACtime
