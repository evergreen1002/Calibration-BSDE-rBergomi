import json
import munch
import os
import logging
import time
from absl import app, flags
from absl import logging as absl_logging
from Solver_Constant import BSDESolver
import numpy as np
import tensorflow as tf


flags.DEFINE_string('config_path', 'rBergomi_const.json', """The path to load json file.""")
flags.DEFINE_string('exp_name', 'test',
                    """The name of numerical experiments, prefix for logging""")
FLAGS = flags.FLAGS
FLAGS.log_dir = './logs' # directory where to write event logs and output array



def main(argv):
    del argv
    with open(FLAGS.config_path) as json_data_file:
        config = json.load(json_data_file)
    config = munch.munchify(config)
    tf.keras.backend.set_floatx(config.net_config.datype)

    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)
    path_prefix = os.path.join(FLAGS.log_dir, FLAGS.exp_name)
    with open('{}_config.json'.format(path_prefix), 'w') as outfile:
        json.dump(dict((name, getattr(config, name))
                  for name in dir(config) if not name.startswith('__')),
        outfile, indent=2)
    absl_logging.get_absl_handler().setFormatter(logging.Formatter('%(levelname)-6s %(message)s'))
    absl_logging.set_verbosity('info')

    logging.info('Begin to solve %s' % config.eqn_config.eqn_name)
    bsde_solver = BSDESolver(config)   
    patience = config.net_config.patience     
    loss_history = []
    aim_history = []
    param_history = []
    wait = 0
    best = float('inf')
    start_time = time.time()

    for i in range(config.net_config.num_iterations):                  
        xi = bsde_solver.model.m_param.xi
        H = bsde_solver.model.m_param.H
        rho = bsde_solver.model.m_param.rho
        eta = bsde_solver.model.m_param.eta 

        with tf.GradientTape() as tape:            
            loss, aim_list = bsde_solver.loss_fn()

        # update all the trainable parameters
        params = bsde_solver.model.trainable_variables       
        grad = tape.gradient(loss, params)
        del tape
        bsde_solver.optimizer_full.apply_gradients(zip(grad, params))
        loss_history.append([loss])        
        aim_history.append(aim_list)
        param_history.append([xi.numpy(), H.numpy(), rho.numpy(), eta.numpy()])

        # The early stopping strategy: stop the training if `aim` does not
        # decrease over a certain number of epochs.
        wait += 1
        if aim_list[0] < best:
            best = aim_list[0]
            wait = 0
        
        if wait >= patience:
            break
    
    np.save("loss_history_5_1.npy", np.array(loss_history))
    np.save("aim_history_5_1.npy", np.array(aim_history)) 
    np.save("param_history_5_1.npy", np.array(param_history))

    
    print("After {i}th iteration".format(i = i+1), "--- %s seconds elapse---" % (time.time() - start_time))

if __name__ == '__main__':
    app.run(main)

 

 