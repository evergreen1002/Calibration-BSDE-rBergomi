import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from KernalApprox import SHIDONGJ_tf



def Cov_SOE_tf(tau, Lambda, H, dtype):
    """
    Covariance matrix for mSOE scheme at time step t_i for i = 1, ..., M-1
    :param tau: time step size
    :param Lambda: nodes of interpolation
    :param H: Hurst index
    return: the covariance matrix with size (N+2, N+2)
    """
    N = len(Lambda)
    cov = tf.TensorArray(dtype, size = 0, dynamic_size = True, clear_after_read = False)
    row_1 = tf.TensorArray(dtype, size = 0, dynamic_size = True, clear_after_read = False)
    # for 1st row
    row_1 = row_1.write(0, tau)
    for i in range(N):
        a = 1/Lambda[i] * (1 - tf.math.exp(-Lambda[i] * tau))
        row_1 = row_1.write(i+1, a)
    row_1 = row_1.write(N+1, tf.math.sqrt(2 * H) / (H + 0.5) * tau **(H + 0.5))
    row_1 = row_1.stack()
    cov = cov.write(0, row_1)

    # for 2nd to N+1 th row
    for j in range(N):
        row_j = tf.TensorArray(dtype, size = 0, dynamic_size = True, clear_after_read = False)
        a = 1/Lambda[j] * (1 - tf.math.exp(-Lambda[j] * tau))
        row_j = row_j.write(0, a)
        for i in range(N):
            lam_sum = Lambda[j] + Lambda[i]
            b = 1/lam_sum * (1 - tf.math.exp(-lam_sum * tau))
            row_j = row_j.write(i+1, b)
        c = tf.math.sqrt(2 * H)/(Lambda[j] ** (H + 0.5)) * tf.math.igamma(H + 0.5, Lambda[j] * tau)\
        * tf.math.exp(tf.math.lgamma(H + 0.5))
        row_j = row_j.write(N+1, c)
        row_j = row_j.stack()
        cov = cov.write(j+1, row_j)
    # for last row
    row_f = tf.TensorArray(dtype, size = 0, dynamic_size = True, clear_after_read = False)
    row_f = row_f.write(0, tf.math.sqrt(2 * H) / (H + 0.5) * tau **(H + 0.5))
    for i in range(N):
        a = tf.math.sqrt(2 * H)/Lambda[i] ** (H + 0.5) * tf.math.igamma(H + 0.5, Lambda[i] * tau)\
        * tf.math.exp(tf.math.lgamma(H + 0.5))
        row_f = row_f.write(i+1, a)
    row_f = row_f.write(N+1, tau**(2*H))
    row_f = row_f.stack()
    cov = cov.write(N+1, row_f)
    cov = cov.stack()
    return cov


def Cov_tf(tau, H, dtype): 
    """
    Covariance matrix for mSOE scheme at time step t_M
    :param tau: time step size    
    :param H: Hurst index
    return: the covariance matrix with size (2, 2)
    """
    cov = tf.TensorArray(dtype, size = 0, dynamic_size = True, clear_after_read = False)
    cov_1_1 = tau 
    cov_1_2 = tf.math.sqrt(H * 2) * tau**(H + 0.5)/ (H + 0.5)
    cov_2_2 = tau**(2 * H)
    row_1 = tf.TensorArray(dtype, size = 0, dynamic_size = True, clear_after_read = False)
    row_1 = row_1.write(0, cov_1_1)
    row_1 = row_1.write(1, cov_1_2)
    row_1 = row_1.stack()
    row_2 = tf.TensorArray(dtype, size = 0, dynamic_size = True, clear_after_read = False)
    row_2 = row_2.write(0, cov_1_2)
    row_2 = row_2.write(1, cov_2_2)
    row_2 = row_2.stack()
    cov = cov.write(0, row_1)
    cov = cov.write(1, row_2)
    cov = cov.stack()
    return cov



class rBergomi_SOE_tf:
    """
    tensorflow version to solve the rBergomi model with time-dependent model parameters by mSOE scheme
    """
    def __init__(self, M, T, P, params):
        #Time discretization
        self.M = M # number of time intervals 
        self.T = T # expiration           
        self.P = P #number of paths to generate
        self.dtype = tf.float64
        self.tau = tf.cast(self.T / self.M, dtype = self.dtype)
        self.grid = tf.cast(tf.linspace(0, T, self.M + 1), dtype = self.dtype)
        self.X0 = tf.cast(params["X0"], dtype = self.dtype)
        self.V0 = tf.cast(params["V0"], dtype = self.dtype)

        #Rough Bergomi model parameters
        self.xi = params["xi"] # tensor of size (1, M)
        self.H = params["H"] # tensor of size (1, M)
        self.rho = params["rho"] # tensor of size (1, M)
        self.nu = params["nu"] # tensor of size (1, M) 

        # Normal distribution
        self.normal_dist = tfp.distributions.Normal(loc = tf.zeros(1, dtype = self.dtype), scale = tf.ones(1, dtype = self.dtype)) 
              
        # Precomputation        
        # size = (1, M)
        self.minue = self.nu**2/2 * tf.reshape(self.grid[1:], [1, -1])**(2 * self.H)

        # Nodes and weights of interpolation        
        Lambda_ = tf.TensorArray(tf.float64, size = 0, dynamic_size = True, clear_after_read = False)
        Omega_ = tf.TensorArray(tf.float64, size = 0, dynamic_size = True, clear_after_read = False)
        for i in range(self.M):            
            soe = SHIDONGJ_tf(0.5 - self.H[0, i], 1e-4, self.tau, self.T)
            lambda_i, omega_i, _ = soe.main()
            Lambda_ = Lambda_.write(i, lambda_i)
            Omega_ = Omega_.write(i, omega_i)

        # size = (M, Nexp)        
        self.Lambda = Lambda_.stack()
        self.Omega = Omega_.stack()
        self.Nexp = self.Lambda.shape[1]

        # Compute \calV(t_i)^j for i = 2,...,M; j = 1,..., Nexp        
        calV = tf.TensorArray(self.dtype, size = 0, dynamic_size = True, clear_after_read = False)
        calV = calV.write(0, tf.zeros(self.Nexp, self.dtype))
        for i in range(1, self.M-1):
            lam_1 = self.Lambda[i, :]
            lam_2 = self.Lambda[i+1, :]
            t = self.grid[i+1]
            row_i = tf.math.sqrt(lam_1/lam_2 * (tf.math.exp(-2 * lam_2 * t) - tf.math.exp(-2 * lam_2 * self.tau))/\
                                 (tf.math.exp(-2 * lam_1 * t) - tf.math.exp(-2 * lam_1 * self.tau)))
            calV = calV.write(i, row_i)
        # size = (M-1, Nexp)
        self.calV = calV.stack()

        
    def multi_dist(self, i):
        """
        i: time step t_{i+1}
        i = 0, ..., M-1
        """
        # compute covariance matrix 
        if i == self.M-1:            
            cov = Cov_tf(self.tau, self.H[0, i], self.dtype)
        else: 
            cov = Cov_SOE_tf(self.tau, self.Lambda[i, :], self.H[0, i], self.dtype)            

        # decomposition, "root" of covariance matrix
        S,U,V = tf.linalg.svd(cov)
        S_half = tf.math.sqrt(S + 1e-15)
        scale = tf.linalg.matmul(tf.linalg.matmul(U, tf.linalg.diag(S_half)), tf.transpose(V)) 
        normal_sample = tf.squeeze(self.normal_dist.sample(sample_shape = [self.P, cov.shape[0]]))

        # size = (self.P, 2) or (P, Nexp + 2)
        return tf.linalg.matmul(normal_sample, scale)


    
    def generate_V(self):   
        """
        Generate volatility paths without multiplying the forward variance curve
        Generate the paths of Brownian motion that drives the stock price 
        """     
        W = []
        mul = []
        hist = tf.zeros((self.P, self.Nexp), dtype = self.dtype)
        sample = self.multi_dist(0)
        W.append(sample[:, 0])
        mul.append(sample[:, -1])
        
        for i in range(self.M - 1):
            # size = (1, Nexp)
            coef = tf.math.exp(-self.tau * tf.reshape(self.Lambda[i+1, :], [1, -1]))
            # size = (P, Nexp)
            hist = (hist * self.calV[i, :]+ sample[:, 1:-1]) * coef
            # size = (P, )
            hist_part = tf.math.sqrt(2*self.H[0, i+1]) * tf.reduce_sum(tf.reshape(self.Omega[i+1], [1, -1]) * hist, axis = 1)
            sample = self.multi_dist(i+1)
            W.append(sample[:, 0])
            mul.append(sample[:, -1] + hist_part)
        
        # size = (P, M)       
        W = tf.stack(W, axis = 1)
        mul = tf.stack(mul, axis = 1)
        V = tf.math.exp(self.nu * mul - self.minue) 
        
        return V, W

    
    def generate_paths(self):
        """
        Generate paths of log stock price
        """
        X = []
        V, W = self.generate_V()
        W_perp = tf.math.sqrt(self.tau) * tf.squeeze(self.normal_dist.sample(sample_shape = [self.P, self.M]))

        # size (P, M)        
        V = self.xi * V
        Z = self.rho * W + tf.math.sqrt(1 - self.rho**2) * W_perp

        # by Forward Euler method，log of stock price        
        X.append((self.X0 - self.V0 * self.tau/2) * tf.ones(self.P, dtype = self.dtype) + tf.math.sqrt(self.V0) * Z[:, 0])

        for j in range(1, self.M):            
            a = X[-1] - V[:, j-1] * self.tau/2 + tf.math.sqrt(V[:, j-1]) * Z[:, j]
            X.append(a)
        X = tf.stack(X, axis = 1)

        return W, W_perp, V, X
    

