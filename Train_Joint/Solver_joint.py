import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from rBergomiSample import rBergomi_SOE_tf
from KernalApprox import SHIDONGJ_tf

def vec(x, num_sample):
    return tf.reshape(x, [num_sample, -1])

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



# class of trainable objects: wrapped by keras layer
class model_param(tf.keras.layers.Layer):
    def __init__(self, config):
        super(model_param, self).__init__()
        self.init_xi = tf.keras.initializers.Constant(value = config.net_config.init_xi)
        self.init_H = tf.keras.initializers.Constant(value = config.net_config.init_H)
        self.init_rho = tf.keras.initializers.Constant(value = config.net_config.init_rho)
        self.init_eta = tf.keras.initializers.Constant(value = config.net_config.init_eta)
        self.xi = self.add_weight(initializer= self.init_xi, trainable= True, name='m_xi')
        self.H = self.add_weight(initializer= self.init_H, trainable= True, name='m_H')
        self.rho = self.add_weight(initializer=self.init_rho, trainable= True, name='m_rho')
        self.eta = self.add_weight(initializer=self.init_eta, trainable= True, name='m_eta')

    def call(self):
        pass



    
class FullModel(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.T = config.eqn_config.T
        self.n = config.eqn_config.n   
        self.tau = self.T/self.n    
        self.dim = config.eqn_config.dim # dim = LxN
        self.L = config.eqn_config.L
        self.N = config.eqn_config.N
        self.k = int(self.n/self.N)
        self.seed = config.eqn_config.seed
        self.datype = config.net_config.datype
        S_0 = tf.cast(config.eqn_config.S0, dtype = self.datype)
        self.X0 = tf.math.log(S_0)
        self.V0 = config.eqn_config.V0
        self.r = tf.cast(config.eqn_config.r, dtype = self.datype)
        self.K = tf.constant(config.eqn_config.K, dtype = self.datype)
        self.valid_size = config.net_config.valid_size        
        self.m_param = model_param(config)
        
        Z_net = []
        t_Z_net = []
        for i in range(0, self.n):
            in_dim = i+2            
            out_dim = int((self.N - int(i/self.k)) * self.L)
            Z_net.append(FeedForwardSubNet(config, in_dim, out_dim))
            t_Z_net.append(FeedForwardSubNet(config, in_dim, out_dim))
    
        self.Z_net = Z_net 
        self.t_Z_net = t_Z_net


    def simulation(self, steps, num_samples):
        tf.random.set_seed(self.seed)
        params = {"X0": self.X0, "V0": self.V0 , "xi": self.m_param.xi, "nu": self.m_param.eta, "rho": self.m_param.rho, "H": self.m_param.H}
        beta = 0.5 - self.m_param.H
        reps = 1e-4        
        soe = SHIDONGJ_tf(beta, reps, self.T/steps, self.T)
        Lambda, Omega, _ = soe.main()
        rbergomi = rBergomi_SOE_tf(steps, self.T, params, num_samples, Lambda, Omega)
        w, w_perp, v, x = rbergomi.generate_paths() 
        return w, w_perp, v, x  
    
    

    def call(self): 
        # Given calibrated model parameters, compute the option prices by mSOE scheme 
        # With time step size 1/1000, MC repetition 10^4
        _, _, _, x = self.simulation(1000, 10000)
        y_theta = []
        h = 1000/self.N
        for j in range(1, self.N+1):
            T_j = self.T/self.N * j             
            payoff = tf.reshape(tf.math.exp(x[:, int(j * h -1)] + self.r * T_j), [-1, 1]) - tf.reshape(self.K, [1, -1])
            payoff = tf.where(payoff < 0, x = 0, y = payoff)            
            y_theta.append(tf.math.exp(-self.r * T_j) * payoff)
        
        y_theta = tf.concat(y_theta, axis = 1)
        # (dim, )
        y_theta = tf.math.reduce_mean(y_theta, axis = 0)
        
        w, w_perp, v, x = self.simulation(self.n, self.valid_size)
        # size (valid_size, n + 1)
        v = tf.concat([tf.ones([self.valid_size, 1], dtype = self.datype) * self.V0, v], axis = 1)
        x = tf.concat([tf.ones([self.valid_size, 1], dtype = self.datype) * self.X0, x], axis = 1)   

        # terminal: list of G_1, G_2, ..., G_N
        # G_j: (valid_size, L)
        terminal = []        
        for j in range(1, self.N+1):
            T_j = self.k * j * self.tau 
            # size (valid_size, L)
            payoff = tf.reshape(tf.math.exp(x[:, int(j * self.k)] + self.r * T_j), [-1, 1]) - tf.reshape(self.K, [1, -1])
            payoff = tf.where(payoff < 0, x = 0, y = payoff)   
            terminal.append(payoff)
        # size (dim)
        y = tf.constant(np.load("rBergomi_TruePrice/TruePrice_{seed}.npy".format(seed = self.seed)), dtype = self.datype) 
        y = tf.reshape(y, [1, -1])
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
                y = (1 + self.r * self.tau) * y + t_z * vec(w[:, i], self.valid_size) + z * vec(w_perp[:, i], self.valid_size)
            loss_j = tf.reduce_mean((terminal[j-1] - y[:, :self.L])**2, axis = 0)
            loss += loss_j
            y = y[:, self.L:]

        # loss
        loss = tf.reduce_mean(loss/ self.N)
        
        return loss, y_theta
    
    
  

class BSDESolver(object):
    def __init__(self, config):
        self.dim = config.eqn_config.dim     
        self.seed = config.eqn_config.seed    
        self.model = FullModel(config)        
        self.valid_size = config.net_config.valid_size
        self.lr_boundaries = config.net_config.lr_boundaries
        self.lr_values = config.net_config.lr_values        
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.lr_boundaries, self.lr_values)        
        self.optimizer_full = tf.keras.optimizers.Adam(learning_rate= lr_schedule, epsilon=1e-10)        

        # size (dim, )        
        self.true_y = tf.constant(np.load("rBergomi_TruePrice/TruePrice_{seed}.npy".format(seed = self.seed)), dtype = self.model.datype) 

    

    def loss_fn(self):        
        loss, y_theta = self.model.call()
        aim =  tf.reduce_sum((y_theta - self.true_y)**2)/self.dim    
        rele_err = tf.math.sqrt(tf.reduce_sum((y_theta - self.true_y)**2)/tf.reduce_sum(self.true_y**2))
        aim_list = [aim, rele_err]

        return loss, aim_list
        