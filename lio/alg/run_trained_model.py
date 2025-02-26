import sys, os
# Add greedy_agent_v1 path FIRST
path_to_add = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, path_to_add)  # This ensures your local version is found first



import tensorflow as tf
import os


from lio.alg import config_room_lio
from lio.env import room_symmetric
from lio.alg.lio_agent import LIO
from lio_agent_exploitative import ExploitativeLIO as LIO_E

# First import the EpisodeLogger class defined earlier
# Assuming it's saved in episode_logger.py
from episode_logger import EpisodeLogger, run_and_log_episode


import lio.alg.lio_agent as lio_module


def load_and_run_trained_model(exp_num=1):
    # Set up the same configuration used in training

    config = config_room_lio.get_config()
    n = 4  # Number of agents in ER(4,2)
    m = 2  # Minimum agents at lever
    config.env.min_at_lever = m
    config.env.n_agents = n
    config.main.dir_name = 'LIO_Exploitative_test_ER42'
    config.main.exp_name = f'er{exp_num}'

    # Create environment
    env = room_symmetric.Env(config.env)

    # Initialize agents
    list_agents = []
    # First agent normal
    agent_0 = LIO(
        config=config.lio,
        l_obs=env.l_obs,
        l_action=env.l_action,
        nn=config.nn,
        agent_name=f'agent_0',
        r_multiplier=config.env.r_multiplier,
        n_agents=env.n_agents,
        agent_id=0
        )
        
    list_agents.append(agent_0)
    
    # Second agent exploitative
    agent_1 = LIO_E(
        config=config.lio,
        l_obs=env.l_obs,
        l_action=env.l_action,
        nn=config.nn,
        agent_name=f'agent_1',
        r_multiplier=config.env.r_multiplier,
        n_agents=env.n_agents,
        agent_id=1
        )
        
    list_agents.append(agent_1)
    
    
    for agent_id in range(2, env.n_agents):
        agent = LIO(
        config=config.lio,
        l_obs=env.l_obs,
        l_action=env.l_action,
        nn=config.nn,
        agent_name=f'agent_{agent_id}',
        r_multiplier=config.env.r_multiplier,
        n_agents=env.n_agents,
        agent_id=agent_id
        )
        
        list_agents.append(agent)

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
    episode_log_dir = os.path.join(exp_dir, 'LIO_Exploitative_testdata_ER42')
    os.makedirs(episode_log_dir, exist_ok=True)

    # Save detailed step log
    logger.save_to_file(os.path.join(episode_log_dir, "test_epsiode_log.csv"))
    
    
    # Generate and save plots
    logger.plot_cumulative_reward(episode_log_dir)
    logger.plot_cumulative_energy(episode_log_dir)
    
    return logger

if __name__ == "__main__":
    logger = load_and_run_trained_model(exp_num=1)