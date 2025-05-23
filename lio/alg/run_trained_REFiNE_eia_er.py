import sys, os
# Add greedy_agent_v1 path FIRST
path_to_add = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, path_to_add)  # This ensures your local version is found first



import tensorflow as tf
import os


from lio.alg import config_room_REFiNE
from lio.env import room_symmetric
from lio.alg.REFiNE_er import REFiNE
from lio.alg.REFiNE_eia_er import REFiNEExploitative as REFiNE_E


from episode_logger_er import run_and_log_episode




# use the trained model in experiment er 1
def load_and_run_trained_model(exp_num=1):
    # Set up the same configuration used in training

    config = config_room_REFiNE.get_config()
    n = 4  # Number of agents in ER(2,1)
    m = 2  # Minimum agents at lever
    config.env.min_at_lever = m
    config.env.n_agents = n
    config.main.dir_name = 'er_REFiNE_attack_4_2'
    config.main.exp_name = f'er{exp_num}'

    # Create environment
    env = room_symmetric.Env(config.env)

    # Initialize agents
    list_agents = []

    # First agent normal
    list_agents.append(REFiNE(config.lio, env.l_obs, env.l_action,config.nn, 'agent_0',config.env.r_multiplier, env.n_agents,0, 1.0))
    
    # Second agent exploitative
    list_agents.append(REFiNE_E(config.lio, env.l_obs, env.l_action,config.nn, 'agent_1',config.env.r_multiplier, env.n_agents,1, 1.0))
    
    for agent_id in range(2, env.n_agents):
        list_agents.append(REFiNE(config.lio, env.l_obs, env.l_action,
                               config.nn, 'agent_%d' % agent_id,
                               config.env.r_multiplier, env.n_agents,
                               agent_id, 1.0))
    

    for agent in list_agents:
        agent.receive_list_of_agents(list_agents)


    # Set up agent networks
    # First create policy gradient ops
    for agent in list_agents:
        agent.create_policy_gradient_op()
        agent.create_update_op()
       
    # Then create reward train ops
    for agent in list_agents:
        agent.create_reward_train_op()


    # Set up TensorFlow session
    config_proto = tf.ConfigProto()
    config_proto.device_count['GPU'] = 0  # Use CPU for inference
    sess = tf.Session(config=config_proto)
    
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    
    # Create saver and restore model
    saver = tf.train.Saver()
    
    # Construct path to saved model
    log_path = os.path.join('..', 'results', config.main.exp_name, config.main.dir_name)
    model_path = os.path.join(log_path, config.main.model_name)
    
    # Restore saved model
    saver.restore(sess, model_path)
    
    # Run episode and log results
    logger = run_and_log_episode(env, list_agents, sess)
    
    # Save logs and generate plots
    exp_dir = os.path.join('..', 'results', config.main.exp_name)
    episode_log_dir = os.path.join(exp_dir, 'test_er_REFiNE_attack_4_2')
    os.makedirs(episode_log_dir, exist_ok=True)

    # Save detailed step log
    logger.save_to_file(os.path.join(episode_log_dir, "test_epsiode_log.csv"))
    
    
    
    
    return logger

if __name__ == "__main__":
    logger = load_and_run_trained_model(exp_num=1)