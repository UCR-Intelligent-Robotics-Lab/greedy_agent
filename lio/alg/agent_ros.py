#!/home/shb/miniconda3/envs/lio/bin python3

import numpy as np
import tensorflow as tf
import random
import os

import argparse
import time 

from lio.alg import config_room_lio
from lio.env import room_symmetric
from lio_agent import LIO


import rospy
from std_msgs.msg import String




sync_flag = False
agent_sync_flag = {}
agent_sync_pub = {}
pub = None
def my_req(data):
    if sync_flag:
        agent_sync_pub[i].publish('req')

def agent1_clback(data):
    agent1_data = data.data
    agent_sync_flag[1] = True

    try:
        msg_data = eval(data.data)  # Parse the message data
        agent1_data = msg_data['data']
        agent_sync_flag[1] = True

        # Log energy metrics from other agent
        if 'total_energy' in msg_data and 'reward_per_energy' in msg_data:
            rospy.loginfo(f"Agent 1 metrics - Total Energy: {msg_data['total_energy']:.3f}, "
                         f"Reward per Energy: {msg_data['reward_per_energy']:.3f}")
    except:
        rospy.logerr("Error parsing agent message")


def agent_sync(agents_num):
    global sync_flag
    sync_flag = True
    other_agents = []
    for i in range(1,agents_num+1):
        agent_sync_flag[i] = False
        # Include energy metrics in sync message
        sync_data = {
            'id': i,
            'data': None,  # Replace with actual agent data
            'total_energy': 0,  # Replace with actual energy
            'reward_per_energy': 0  # Replace with actual metric
        }
        agent_sync_pub[i].publish(json.dumps(sync_data))

        while not agent_sync_flag[i]:
            time.sleep(0.01)
        
        other_agents.append(data)
    
    sync_flag = False
    return other_agents

def calculate_reward_per_energy(buffer):
    """Calculate reward per energy using only environmental rewards"""
    if buffer.total_energy > 0:
        env_rewards = sum(buffer.reward)
        return env_rewards / buffer.total_energy
    return 0

class Buffer(object):
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.reset()

    def reset(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.obs_next = []
        self.done = []
        self.r_from_others = []
        self.r_given = []
        self.action_all = []
        self.energy_cost = []  # Stores energy costs per step
        self.total_energy = 0  # Stores total energy consumed by the agent

    def add(self, transition, energy):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.obs_next.append(transition[3])
        self.done.append(transition[4])
        self.energy_cost.append(energy)
        self.total_energy += energy

    def add_r_from_others(self, r):
        self.r_from_others.append(r)

    def add_action_all(self, list_actions):
        self.action_all.append(list_actions)

    def add_r_given(self, r):
        self.r_given.append(r)       

def get_agent_list_buffer():
    pass
def get_list_buffers_new():
    pass
def init_agent():
    seed = config.main.seed
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

    dir_name = config.main.dir_name
    exp_name = config.main.exp_name
    log_path = os.path.join('..', 'results', exp_name, dir_name)
    model_name = config.main.model_name
    save_period = config.main.save_period


    n_episodes = int(config.alg.n_episodes)
    n_eval = config.alg.n_eval
    period = config.alg.period

    epsilon = config.lio.epsilon_start
    epsilon_step = (
        epsilon - config.lio.epsilon_end) / config.lio.epsilon_div


    env = room_symmetric.Env(config.env) # TODO: remove

    agent = LIO(config.lio, env.l_obs, env.l_action,
                            config.nn, 'agent_0',
                             config.env.r_multiplier, env.n_agents, energy_param=1.0))

    other_agents = agent_sync()
    agent.create_policy_gradient_op(other_agents)#
    agent.create_update_op()

    other_agents = agent_sync()
    agent.create_reward_train_op(other_agents)#


    config_proto = tf.ConfigProto()
    if config.main.use_gpu:
        config_proto.device_count['GPU'] = 1
        config_proto.gpu_options.allow_growth = True
    else:
        config_proto.device_count['GPU'] = 0
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())

    list_agent_meas = []
    if config.env.name == 'er':
         list_suffix = ['reward_total', 'rewards_env','n_lever', 'n_door',
                       'received', 'given', 'r-lever', 'r-start', 'r-door', 'mission_status', 
                              'total_energy', 'reward_per_energy']

    for agent_id in range(1, env.n_agents + 1):
        for suffix in list_suffix:
            list_agent_meas.append('A%d_%s' % (agent_id, suffix))

    saver = tf.train.Saver(max_to_keep=config.main.max_to_keep)

    # Training loop with energy tracking
    agent_sync()

    list_buffers = get_agent_list_buffer()
    # Track energy metrics before update
    pre_total_energy = list_buffers[0].total_energy
    pre_reward_per_energy = calculate_reward_per_energy(list_buffers[0])
    agent.update(sess, list_buffers, epsilon)

    other_agents = agent_sync()
    list_buffers_new = get_list_buffers_new()
    if agent.can_give:
        agent.train_reward(sess, list_buffers,
                            list_buffers_new, epsilon, other_agents)#

    agent_sync()

    agent.update_main(sess)

    # Log energy metrics after update
    post_total_energy = list_buffers[0].total_energy
    post_reward_per_energy = calculate_reward_per_energy(list_buffers[0])

    # Log energy changes
    rospy.loginfo(f"Agent Energy Metrics - Before Update: "
                  f"Total Energy: {pre_total_energy:.3f}, "
                  f"Reward/Energy: {pre_reward_per_energy:.3f}")
    rospy.loginfo(f"Agent Energy Metrics - After Update: "
                  f"Total Energy: {post_total_energy:.3f}, "
                  f"Reward/Energy: {post_reward_per_energy:.3f}")

def talker():
    rate = rospy.Rate(100) # 100hz
    while not rospy.is_shutdown():
        time_str = f"Status update {rospy.get_time()}"
        rospy.loginfo(time_str)
        pub.publish(time_str)
        rate.sleep()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('id', type=int)
    args = parser.parse_args()
    try:

        rospy.init_node("agent%d"%(args.id), anonymous=False)
        pub = rospy.Publisher('chatter', String, queue_size=10)
        rospy.Subscriber(f"agent{args.id}/req", String, my_req)
        rospy.Subscriber("agent1/data", String, agent1_clback)

         # Initialize publisher dictionary
        for i in range(1, env.n_agents + 1):
            agent_sync_pub[i] = rospy.Publisher(f'agent{i}/sync', String, queue_size=10)
        config = config_room_lio.get_config()
        init_agent()
        talker()
    except rospy.ROSInterruptException:
        pass