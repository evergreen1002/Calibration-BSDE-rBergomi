import numpy as np
import tensorflow as tf
from KernalApprox import SHIDONGJ_tf
from rBergomiSample import rBergomi_SOE_tf

def European_price(model_param, seed): 
    # model_param = [xi, H, rho, nu]
    M = 1000
    T = 1
    tau = tf.cast(T/M, dtype = tf.float64)
    X0 = tf.cast(tf.math.log(100.0), dtype = tf.float64)
    V0 = tf.cast(0.09, dtype = tf.float64)
    xi = tf.cast(model_param[0], dtype = tf.float64)
    H = tf.cast(model_param[1], dtype = tf.float64)
    rho = tf.cast(model_param[2], dtype = tf.float64)
    nu = tf.cast(model_param[3], dtype = tf.float64)
    r = tf.cast(0.05, dtype = tf.float64)    
    P = int(10000)
    N = 5 
    L= 9    
    params = {"X0": X0, "V0": V0, "xi": xi, "nu": nu, "rho": rho, "H": H}
    beta = 0.5 - H
    reps = tf.cast(1e-4, dtype = tf.float64)

    soe_tf = SHIDONGJ_tf(beta, reps, tau, T)
    Lambda_tf, Omega_tf, _ = soe_tf.main()
    rbergomi_tf = rBergomi_SOE_tf(M, T, params, P, Lambda_tf, Omega_tf)

    tf.random.set_seed(seed)
    _, _, _, X_1 = rbergomi_tf.generate_paths()

    K = tf.constant([50, 55, 60, 65, 70, 75, 80, 85, 90], dtype = tf.float64)
    y_theta = []
    h = 1000/N
    for j in range(1, N+1):
        T_j = T/N * j
        # size (valid_size, L)
        payoff = tf.reshape(tf.math.exp(X_1[:, int(j * h -1)] + r * T_j), [-1, 1]) - tf.reshape(K, [1, -1])
        payoff = tf.where(payoff < 0, x = 0, y = payoff)            
        y_theta.append(tf.math.exp(-r * T_j) * payoff)

    # size (10000, 45)     
    y_theta = tf.concat(y_theta, axis = 1)
    # size (45, )
    price = tf.math.reduce_mean(y_theta, axis = 0)
    return price

"""
model_param = [0.09, 0.07, -0.9, 1.9]
seed = [512]
for i in range(5):
    my_seed = seed[i]
    price = European_price(model_param, seed = my_seed)
    np.save("rBergomi_TruePrice/TruePrice_{i}".format(i = my_seed), price.numpy())
"""















