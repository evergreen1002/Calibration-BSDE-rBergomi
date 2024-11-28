import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from rBergomiSample_functionall import rBergomi_SOE_tf
from KernalApprox import SHIDONGJ_tf
import tf_quant_finance as tff
from scipy import integrate


def find_nearest(a, a_val):
# return the element in a that is nearest to a_val
   idx = np.abs(a - a_val).argmin()
   return a[idx], idx

def vec(x, num_sample):
    return tf.reshape(x, [num_sample, -1])


# network for xi, H, rho, eta
class FCNN(tf.keras.Model):
    def __init__(self, config, init):        
        super().__init__()
        self.num_hiddens = config.param_config.num_hiddens
        self.dense_layers = [tf.keras.layers.Dense(self.num_hiddens[i],
                                                   use_bias=True,
                                                   kernel_initializer = tf.initializers.Constant(value = 0.01), 
                                                   bias_initializer = tf.initializers.Constant(value = 0.0), 
                                                   activation=tf.nn.leaky_relu)
                             for i in range(len(self.num_hiddens))]
        self.dense_layers.append(tf.keras.layers.Dense(1, 
                                                       kernel_initializer = tf.initializers.Constant(value = -0.02), 
                                                       bias_initializer = tf._initializers.Constant(value = init),
                                                       activation = None))

    def call(self, t):
        t = tf.reshape(t, [-1, 1])
        for i in range(len(self.dense_layers)):
            t = self.dense_layers[i](t)        
        return t
    

# subnet for Z and \tilde{Z}
class FeedForwardSubNet(tf.keras.Model):
    # in_dim: input dimension 
    # out_dim: output_dimension     
    def __init__(self, config, in_dim, out_dim):
        super().__init__()  
        self.in_dim = in_dim
        self.num_hiddens = config.net_config.num_hiddens
        
        # total (num_hiddens + 2) bn layers 
        self.bn_layers = [
        tf.keras.layers.BatchNormalization(
            momentum = 0.99,
            epsilon = 1e-6,
            beta_initializer = tf.random_normal_initializer(0, stddev = 0.1),
            gamma_initializer = tf.random_uniform_initializer(0.1, 0.5)
        )
        for _ in range(len(self.num_hiddens) + 2)
        ]
        
        # total (num_hiddens + 1) dense layers        
        self.dense_layers = [tf.keras.layers.Dense(self.num_hiddens[i], use_bias = True, activation = None) for i in range(len(self.num_hiddens))]
        
        # output = Z or \tilde{Z} 
        self.dense_layers.append(tf.keras.layers.Dense(out_dim, activation = None))


    def call(self, x):        
        # structure: bn -> (dense -> drouput -> bn -> leaky_relu) * len(num_hiddens) -> dense -> drouput -> bn
        x = tf.reshape(x, [-1, self.in_dim])
        x = self.bn_layers[0](x, training = True)
        for i in range(len(self.num_hiddens)):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training = True)
            x = tf.nn.leaky_relu(x, alpha = 0.3)        
        x = self.dense_layers[-1](x)        
        x = self.bn_layers[-1](x, training = True)        
        return x


    
class FullModel(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.T = config.eqn_config.T
        self.n = config.eqn_config.n   
        self.tau = self.T/self.n
        self.L = config.eqn_config.L
        self.N = config.eqn_config.N
        self.dim = self.L * self.N
        self.k = int(self.n/self.N)
        self.seed = config.eqn_config.seed
        self.datype = config.net_config.datype
        S_0 = tf.cast(config.eqn_config.S0, dtype = self.datype)
        self.X0 = tf.math.log(S_0)
        self.V0 = config.eqn_config.V0        
        self.K = tf.constant(config.eqn_config.K, dtype = tf.float64)
        self.valid_size = config.net_config.valid_size   

        # model parameter as a NN
        self.xi_NN = FCNN(config, init = config.param_config.init_xi)    
        self.H_NN = FCNN(config, init = config.param_config.init_H)
        self.rho_NN = FCNN(config, init = config.param_config.init_rho)
        self.eta_NN = FCNN(config, init = config.param_config.init_eta)   

        Z_net = []
        t_Z_net = []
        for i in range(0, self.n):
            in_dim = i+2            
            out_dim = int((self.N - int(i/self.k)) * self.L)
            Z_net.append(FeedForwardSubNet(config, in_dim, out_dim))
            t_Z_net.append(FeedForwardSubNet(config, in_dim, out_dim))    
        self.Z_net = Z_net 
        self.t_Z_net = t_Z_net

        # find N values in T_tar that is nearest to the grid 
        self.grid = tf.linspace(0, 1, self.N+1)[1:]
        T_tar = np.load("Dataset/Maturities_49.npy")
        self.T_hist = np.zeros(self.N)
        self.idx = np.zeros(self.N)
        for i in range(self.N):
            self.T_hist[i], self.idx[i] = find_nearest(T_tar, self.grid[i].numpy())
        self.T_hist = tf.constant(self.T_hist, dtype = tf.float64)

        # cubic spline interpolation for interest rate
        r_T = np.load("Dataset/Interest_rate_date_9.npy")
        r = np.load("Dataset/Interest_rate_9.npy")        
        self.spline = tff.math.interpolation.cubic.build_spline(r_T, r)
        # size (n+1, )
        fine_grid = tf.linspace(0, 1, self.n+1)
        self.r = tff.math.interpolation.cubic.interpolate(fine_grid, self.spline)/100
    
    # compute the integral \int_0^t r(s)ds
    def integ_r(self, t):
        f = lambda x: (tff.math.interpolation.cubic.interpolate(np.array([x]), self.spline)/100).numpy()[0]
        result = integrate.quad(f, 0.0, t)[0]
        return result
    
    # compute the model parameters on a give grid 
    def call_m(self, m_NN, steps):
        return tf.reshape(m_NN.call(tf.linspace(0, self.T, steps + 1)[1:]), [1, -1]) 

    def simulation(self, steps, num_samples): 
        xi = self.call_m(self.xi_NN, steps)
        H = self.call_m(self.H_NN, steps)
        rho = self.call_m(self.rho_NN, steps)
        eta = self.call_m(self.eta_NN, steps)
        model_params = [xi, H, rho, eta]
                  
        tf.random.set_seed(self.seed) 
        params = {"X0": self.X0, "V0": self.V0 , "xi": xi, "H": H, "rho": rho, "nu": eta}        
        rbergomi = rBergomi_SOE_tf(steps, self.T, num_samples, params)
        W, W_perp, V, X = rbergomi.generate_paths()
        
        return W, W_perp, V, X, model_params

    

    def call(self):
        # Given calibrated model parameters, compute the interpolated option prices by mSOE scheme 
        # With time step size 1/1000, MC repetition 10^4              
        _, _, _, x, model_params_full = self.simulation(1000, 10000)
        y_aim = []
        h = 1000/self.N
        for j in range(1, self.N+1):
            T_j = self.T/self.N * j 
            int_r = tf.cast(self.integ_r(T_j), tf.float64)
            payoff = tf.reshape(tf.math.exp(x[:, int(j * h -1)] + int_r), [-1, 1]) - tf.reshape(self.K, [1, -1])
            payoff = tf.where(payoff < 0, x = 0, y = payoff)            
            y_aim.append(tf.math.exp(-int_r) * payoff)
        
        y_aim = tf.concat(y_aim, axis = 1)
        aim = tf.reshape(tf.math.reduce_mean(y_aim, axis = 0), [self.N, self.L])

        # cubic spline interpolation on T
        aim_interp = tf.TensorArray(tf.float64, size = 0, dynamic_size = True, clear_after_read = False)           
        for i in range(self.L):
            spline = tff.math.interpolation.cubic.build_spline(self.grid, aim[:, i])
            price_i = tff.math.interpolation.cubic.interpolate(self.T_hist, spline)
            aim_interp = aim_interp.write(i, price_i)
        
        # size (NxL, )
        aim_interp = tf.reshape(tf.transpose(aim_interp.stack()), [-1])
        

        w, w_perp, v, x, _= self.simulation(self.n, self.valid_size)
        # size (valid_size, n + 1)
        v = tf.concat([tf.ones([self.valid_size, 1], dtype = tf.float64) * self.V0, v], axis = 1)
        x = tf.concat([tf.ones([self.valid_size, 1], dtype = tf.float64) * self.X0, x], axis = 1)

        
        # terminal: list of y_1, y_2, ..., y_N
        terminal = []        
        for j in range(1, self.N+1):
            idx = int(j * self.k)
            T_j = self.k * j * self.tau 
            # size (valid_size, L)
            payoff = tf.reshape(tf.math.exp(x[:, idx] + self.integ_r(T_j)), [-1, 1]) - tf.reshape(self.K, [1, -1])
            payoff = tf.where(payoff < 0, x = 0, y = payoff)   
            terminal.append(payoff)
        
        
        price_tar = np.load("Dataset/Option_prices_49_15.npy")        
        # size (N, L)
        y = tf.constant(price_tar[self.idx.astype(np.int32), :self.L], dtype = tf.float64)
         
        # cubic spline interpolation on T, so that y falls on the equidistant grid
        y_interp = tf.TensorArray(tf.float64, size = 0, dynamic_size = True, clear_after_read = False)           
        for i in range(self.L):
            spline = tff.math.interpolation.cubic.build_spline(self.T_hist, y[:, i])
            price_i = tff.math.interpolation.cubic.interpolate(self.grid, spline)
            y_interp = y_interp.write(i, price_i)
        
        # size (dim, )
        y = tf.reshape(tf.transpose(y_interp.stack()), [-1])
        loss = 0
        for j in range(1, self.N+1):
            id_end = j * self.k
            id_start = (j-1)* self.k
            for i in range(id_start, id_end):
                # size (valid_size,  i+2)
                input = tf.concat([vec(v[:, :i+1], self.valid_size), vec(x[:, i], self.valid_size)], axis = 1)                
                z = self.Z_net[i].call(input)
                t_z = self.t_Z_net[i].call(input)
                # size (valid_size, (N+1 - j)*L)
                y = (1 + self.r[i] * self.tau) * y + t_z * vec(w[:, i], self.valid_size) + z * vec(w_perp[:, i], self.valid_size)
            loss_j = tf.reduce_mean((terminal[j-1] - y[:, :self.L])**2, axis = 0)
            loss += loss_j
            y = y[:, self.L:]

        # loss
        loss = tf.reduce_mean(loss/ self.N)
        return loss, aim_interp, model_params_full
    
  

class BSDESolver(object):
    def __init__(self, config):
        self.model = FullModel(config)    
        self.dim = self.model.dim    
        self.valid_size = config.net_config.valid_size
        self.lr_boundaries = config.net_config.lr_boundaries
        self.lr_values = config.net_config.lr_values        
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.lr_boundaries, self.lr_values)        
        self.optimizer_full = tf.keras.optimizers.Adam(learning_rate= lr_schedule, epsilon=1e-10)        

        # size (dim, )
        price_tar = np.load("Dataset/Option_prices_49_15.npy")        
        idx = self.model.idx
        L = self.model.L
        self.true_y = tf.reshape(tf.constant(price_tar[idx.astype(np.int32), :L], dtype = tf.float64), [-1])

    def rele_err(self, y):        
        err = np.abs(self.true_y - y)/self.true_y
        max_err = np.max(err)
        avg_err = np.mean(err)
        return max_err, avg_err


    def loss_fn(self):        
        loss, y_theta, model_params = self.model.call()
        aim =  tf.reduce_sum((y_theta - self.true_y)**2)/self.dim    
        aim_max_err, aim_avg_err = self.rele_err(y_theta) 
        aim_list = [aim, aim_max_err, aim_avg_err]

        return loss, aim_list, model_params
    