import json
import munch
import os
import logging
import time
from absl import app, flags
from absl import logging as absl_logging
from Solver_joint import BSDESolver
import numpy as np
import tensorflow as tf


flags.DEFINE_string('config_path', 'rBergomi.json', """The path to load json file.""")
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
     
    training_history = []
    aim_history = []
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
        training_history.append([i+1, loss.numpy(), xi.numpy(), H.numpy(), rho.numpy(), eta.numpy()])
        aim_history.append(aim_list) 
        
        print("Step: {i}, Loss: {loss}, xi: {xi}, H: {H}, rho: {rho}, eta : {eta}".format(i = i+1, 
                                                                            loss=loss.numpy(), xi=xi.numpy(),                                                                                                                                                                    
                                                                            H=H.numpy(), rho=rho.numpy(), eta=eta.numpy()))  
        

        np.save("Convergence_20/training_history_20_1.npy", np.array(training_history))
        np.save("Convergence_20/y_20_1.npy", np.array(aim_history)) 
        print("--- %s seconds elapse---" % (time.time() - start_time))
    
    

if __name__ == '__main__':
    app.run(main)

 

 