"""Meta Learning to Incentivize Others (Meta-LIO) implementation.
Extends LIO with meta-learning components."""

import numpy as np
import tensorflow as tf
from lio.alg import networks
import lio.utils.util as util

class MetaLIO(object):
    """Meta-LIO built on policy gradient."""
    
    def __init__(self, config, l_obs, l_action, nn, agent_name,
                 r_multiplier=2, n_agents=1, agent_id=0, energy_param=1.0):
        """Initialize Meta-LIO agent.
        
        Args:
            config: Configuration object
            l_obs: Length of observation
            l_action: Length of action space
            nn: Neural network parameters
            agent_name: Name of agent
            r_multiplier: Reward multiplier
            n_agents: Number of agents
            agent_id: ID of this agent
            energy_param: Parameter scaling energy consumption
        """
        self.alg_name = 'meta-lio'
        self.l_obs = l_obs
        self.l_action = l_action
        self.nn = nn
        self.agent_name = agent_name
        self.r_multiplier = r_multiplier
        self.n_agents = n_agents
        self.agent_id = agent_id
        self.energy_param = energy_param  # New parameter for energy

        # Meta-learning parameters
        # Learning rate for meta-optimization
        self.meta_lr = config.meta_lr if hasattr(config, 'meta_lr') else 0.001
        # Batch size for meta-learning updates
        self.meta_batch_size = config.meta_batch_size if hasattr(config, 'meta_batch_size') else 32

        # Core LIO parameters
        self.entropy_coeff = config.entropy_coeff
        self.gamma = config.gamma
        self.include_cost_in_chain_rule = config.include_cost_in_chain_rule
        self.lr_actor = config.lr_actor
        self.lr_cost = config.lr_cost
        self.lr_reward = config.lr_reward
        if 'optimizer' in config:
            self.optimizer = config.optimizer
        else:
            self.optimizer = 'sgd'
        self.reg = config.reg
        self.reg_coeff = config.reg_coeff
        self.separate_cost_optimizer = config.separate_cost_optimizer

        assert not (self.separate_cost_optimizer and self.include_cost_in_chain_rule)
        
        self.list_other_id = list(range(0, self.n_agents))
        del self.list_other_id[self.agent_id]
        self.can_give = True

        # Create networks and meta networks
        self.create_networks() # This creates policy and reward networks
        self.create_meta_networks() # This creates meta-policy and meta-reward networks
        self.policy_new = PolicyNew # For policy updates
        print(f"Initializing LIO_Meta agent {self.agent_name} with energy_param: {energy_param}")


    def create_networks(self):
        """Create primary policy and reward networks."""
        self.obs = tf.placeholder(tf.float32, [None, self.l_obs], 'l_obs')
        self.action_others = tf.placeholder(
            tf.float32, [None, self.l_action * (self.n_agents - 1)])
        self.epsilon = tf.placeholder(tf.float32, None, 'epsilon')

        with tf.variable_scope(self.agent_name):
            with tf.variable_scope('policy_main'):
                with tf.variable_scope('policy'):
                    probs = networks.actor_mlp(self.obs, self.l_action, self.nn)
                with tf.variable_scope('eta'):
                    self.reward_function = networks.reward_mlp(
                        self.obs, self.action_others, self.nn,
                        n_recipients=self.n_agents)
                    
                self.probs = (1 - self.epsilon) * probs + self.epsilon / self.l_action
                self.log_probs = tf.log(self.probs)
                self.action_samples = tf.multinomial(self.log_probs, 1)

            with tf.variable_scope('policy_prime'):
                with tf.variable_scope('policy'):
                    probs = networks.actor_mlp(self.obs, self.l_action, self.nn)
                self.probs_prime = (1-self.epsilon)*probs + self.epsilon/self.l_action
                self.log_probs_prime = tf.log(self.probs_prime)
                self.action_samples_prime = tf.multinomial(self.log_probs_prime, 1)

        self.policy_params = tf.trainable_variables(
            self.agent_name + '/policy_main/policy')
        self.policy_prime_params = tf.trainable_variables(
            self.agent_name + '/policy_prime/policy')

        self.list_copy_main_to_prime_ops = []
        for idx, var in enumerate(self.policy_prime_params):
            self.list_copy_main_to_prime_ops.append(
                var.assign(self.policy_params[idx]))

        self.list_copy_prime_to_main_ops = []
        for idx, var in enumerate(self.policy_params):
            self.list_copy_prime_to_main_ops.append(
                var.assign(self.policy_prime_params[idx]))
            
    def receive_list_of_agents(self, list_of_agents):
        self.list_of_agents = list_of_agents

    def calculate_energy_cost(self, state, action):
    # This function calculates the energy consumed based on the current state, action, and agent-specific energy parameter
        return self.energy_param * np.abs(action) * (1 + state.mean())        

    def create_meta_networks(self):
        """Create meta-learning networks."""
        with tf.variable_scope(self.agent_name):
            with tf.variable_scope('meta'):
                # Meta network for policy adaptation
                self.meta_policy = networks.actor_mlp(
                    self.obs, self.l_action, self.nn)
                
                # Meta network for reward function adaptation
                self.meta_reward = networks.reward_mlp(
                    self.obs, self.action_others, self.nn,
                    n_recipients=self.n_agents)

        # Get meta parameters
        self.meta_params = tf.trainable_variables(
            self.agent_name + '/meta')
        
    def create_energy_networks(self):
        """Create energy tracking networks without optimization."""
        # Energy consumption network
        with tf.variable_scope(self.agent_name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope('energy', reuse=tf.AUTO_REUSE):
                self.energy_net = tf.layers.dense(
                    self.obs, 1, activation=tf.nn.relu,
                    name='energy_consumption')

        # Energy parameters (for tracking only)
        self.energy_params = tf.trainable_variables(
            self.agent_name + '/energy')

        # Placeholders for energy calculations
        self.avg_energy = tf.placeholder(tf.float32, None, name=f'avg_energy_{self.agent_id}')
        self.avg_reward_per_energy = tf.placeholder(tf.float32, None, name=f'avg_reward_per_energy_{self.agent_id}')
        self.total_rewards_placeholder = tf.placeholder(tf.float32, [None], name=f'total_rewards_{self.agent_id}')

        # Calculate individual energy metrics
        self.total_energy = tf.reduce_sum(self.energy_net)
        self.reward_per_energy = tf.reduce_sum(tf.abs(self.total_rewards_placeholder)) / \
                               (self.total_energy + 1e-8)

    
    def create_meta_objective(self):
        """Create the meta-learning objective."""
        # Get inputs for meta-learning
        self.meta_returns = tf.placeholder(tf.float32, [None], 'meta_returns')

        # Add mask placeholder
        self.mask = tf.placeholder(tf.float32, [None], 'mask')

        # Get meta policy outputs(In ER case: action probabilities for lever/start/door)
        meta_policy_probs = tf.nn.softmax(self.meta_policy)
        
        # Calculate meta policy loss 
        meta_log_probs = tf.log(meta_policy_probs + 1e-15)
        # Get batch size from the actual tensor
        batch_size = tf.shape(self.meta_returns)[0]
        # Reshape meta_returns to [batch_size, 1] then tile to match action dimension
        meta_returns_4policy = tf.tile(
            tf.reshape(self.meta_returns, [-1, 1]), # Reshape to [batch_size, 1]
            [1, tf.shape(meta_log_probs)[1]]
        )

        # Apply mask to policy weighted sum
        meta_policy_weighted = tf.reduce_sum(
            tf.multiply(meta_log_probs, meta_returns_4policy),
            axis=1)  # Sum over action dimension
        self.meta_policy_loss = -tf.reduce_mean(meta_policy_weighted)

        # Calculate meta reward loss with masking
        # Get batch size dynamically
        reward_shape = tf.shape(self.meta_reward)
        n_outputs = reward_shape[1]

        # Reshape meta_returns to match the actual reward output shape
        meta_returns_4reward = tf.tile(
            tf.reshape(self.meta_returns[:batch_size], [-1, 1]),
            [1, n_outputs])  # Repeat for each other agent
        
        # Ensure meta_reward uses same batch size
        meta_reward_batch = self.meta_reward[:batch_size]
        
        # Calculate reward loss directly without reshaping the outputs
        self.meta_reward_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.square(meta_reward_batch - meta_returns_4reward),
                axis=1))  # Sum over other agents

        
        # Total meta loss combines both components
        self.meta_loss = self.meta_policy_loss + self.meta_reward_loss
        # Create masked version for training
        self.masked_meta_loss = self.meta_loss * tf.reduce_mean(self.mask)
        # Create meta optimizer - FIXED
        self.meta_opt = tf.train.AdamOptimizer(self.meta_lr)
        # Use masked_meta_loss for training
        self.meta_train_op = self.meta_opt.minimize(
            self.masked_meta_loss, var_list=self.meta_params)
        
        
        
        
    def run_actor(self, obs, sess, epsilon, prime=False):
        """Run actor to select actions."""
        feed = {self.obs: np.array([obs]), self.epsilon: epsilon}
        if prime:
            action = sess.run(self.action_samples_prime, feed_dict=feed)[0][0]
        else:
            action = sess.run(self.action_samples, feed_dict=feed)[0][0]
        # Calculate energy consumption for this action
        energy_cost = self.calculate_energy_cost(obs, action)      
        return action

    def give_reward(self, obs, action_all, sess):
        """Give reward to other agents."""
        action_others_1hot = util.get_action_others_1hot(
            action_all, self.agent_id, self.l_action)
        feed = {
            self.obs: np.array([obs]),
            self.action_others: np.array([action_others_1hot])
        }
        reward = sess.run(self.reward_function, feed_dict=feed)
        reward = reward.flatten() * self.r_multiplier
        return reward
    
    def create_policy_gradient_op(self):
        self.r_ext = tf.placeholder(tf.float32, [None], 'r_ext')

        r2 = self.r_ext
        this_agent_1hot = tf.one_hot(indices=self.agent_id, depth=self.n_agents)
        for other_id in self.list_other_id:
            r2 += self.r_multiplier * tf.reduce_sum(
                tf.multiply(self.list_of_agents[other_id].reward_function,
                            this_agent_1hot), axis=1)

        if self.include_cost_in_chain_rule:
            # for this agent j, subtract the rewards given to all other agents
            # i.e. minus \sum_{i=1}^{N-1} r^i_{eta^j}
            reverse_1hot = 1 - tf.one_hot(indices=self.agent_id, depth=self.n_agents)
            r2 -= self.r_multiplier * tf.reduce_sum(
                tf.multiply(self.reward_function, reverse_1hot), axis=1)

        self.ones = tf.placeholder(tf.float32, [None], 'ones')
        self.gamma_prod = tf.math.cumprod(self.ones * self.gamma)
        returns = tf.reverse(
            tf.math.cumsum(tf.reverse(r2 * self.gamma_prod, axis=[0])), axis=[0])
        returns = returns / self.gamma_prod

        self.action_taken = tf.placeholder(tf.float32, [None, self.l_action],
                                           'action_taken')
        self.log_probs_taken = tf.log(tf.reduce_sum(
            tf.multiply(self.probs, self.action_taken), axis=1) + 1e-15)

        self.entropy = -tf.reduce_sum(tf.multiply(self.probs, self.log_probs))

        self.policy_loss = -tf.reduce_sum(
            tf.multiply(self.log_probs_taken, returns))
        self.loss = self.policy_loss - self.entropy_coeff * self.entropy

        self.policy_grads = tf.gradients(self.loss, self.policy_params)
        grads_and_vars = list(zip(self.policy_grads, self.policy_params))
        self.policy_opt = tf.train.GradientDescentOptimizer(self.lr_actor)
        self.policy_op = self.policy_opt.apply_gradients(grads_and_vars)

    
        


    def create_update_op(self):
        self.r_from_others = tf.placeholder(tf.float32, [None], 'r_from_others')
        r2_val = self.r_ext + self.r_from_others
        if self.include_cost_in_chain_rule:
            self.r_given = tf.placeholder(tf.float32, [None], 'r_given')
            r2_val -= self.r_given
        returns_val = tf.reverse(tf.math.cumsum(
            tf.reverse(r2_val * self.gamma_prod, axis=[0])), axis=[0])
        returns_val = returns_val / self.gamma_prod

        log_probs_taken = tf.log(tf.reduce_sum(
            tf.multiply(self.probs_prime, self.action_taken), axis=1) + 1e-15)
        entropy = -tf.reduce_sum(
            tf.multiply(self.probs_prime, self.log_probs_prime))
        policy_loss = -tf.reduce_sum(
            tf.multiply(log_probs_taken, returns_val))
        loss = policy_loss - self.entropy_coeff * entropy

        policy_opt_prime = tf.train.GradientDescentOptimizer(self.lr_actor)
        self.policy_op_prime = policy_opt_prime.minimize(loss)

    def create_reward_train_op(self):
        list_reward_loss = []
        self.list_policy_new = [0 for x in range(self.n_agents)]
        self.returns = tf.placeholder(tf.float32, [None], 'returns')

        for agent in self.list_of_agents:
            if agent.agent_id == self.agent_id and not self.include_cost_in_chain_rule:
                # In this case, cost for giving is not accounted in chain rule,
                # so the agent can skip over itself
                continue
            other_policy_params_new = {}
            for grad, var in zip(agent.policy_grads, agent.policy_params):
                other_policy_params_new[var.name] = var - agent.lr_actor * grad
            other_policy_new = agent.policy_new(
                other_policy_params_new, agent.l_obs, agent.l_action,
                agent.agent_name)
            self.list_policy_new[agent.agent_id] = other_policy_new

            log_probs_taken = tf.log(
                tf.reduce_sum(tf.multiply(other_policy_new.probs,
                                          other_policy_new.action_taken), axis=1))
            loss_term = -tf.reduce_sum(tf.multiply(log_probs_taken, self.returns))
            list_reward_loss.append(loss_term)

        if self.include_cost_in_chain_rule:
            self.reward_loss = tf.reduce_sum(list_reward_loss)
        else:
            reverse_1hot = 1 - tf.one_hot(indices=self.agent_id, depth=self.n_agents)
            if self.separate_cost_optimizer or self.reg == 'l1':
                given_each_step = tf.reduce_sum(tf.abs(
                    tf.multiply(self.reward_function, reverse_1hot)), axis=1)
                total_given = tf.reduce_sum(tf.multiply(
                    given_each_step, self.gamma_prod/self.gamma))
            elif self.reg == 'l2':
                total_given = tf.reduce_sum(tf.square(
                    tf.multiply(self.reward_function, reverse_1hot)))
            if self.separate_cost_optimizer:
                self.reward_loss = tf.reduce_sum(list_reward_loss)
            else:
                self.reward_loss = (tf.reduce_sum(list_reward_loss) +
                                    self.reg_coeff * total_given)
            
        if self.optimizer == 'sgd':
            reward_opt = tf.train.GradientDescentOptimizer(self.lr_reward)
            if self.separate_cost_optimizer:
                cost_opt = tf.train.GradientDescentOptimizer(self.lr_cost)
        elif self.optimizer == 'adam':
            reward_opt = tf.train.AdamOptimizer(self.lr_reward)
            if self.separate_cost_optimizer:
                cost_opt = tf.train.AdamOptimizer(self.lr_cost)
        self.reward_op = reward_opt.minimize(self.reward_loss)
        if self.separate_cost_optimizer:
            self.cost_op = cost_opt.minimize(total_given)


        
    def update(self, sess, buf, epsilon):
        """Update policy parameters with meta-learning."""
        sess.run(self.list_copy_main_to_prime_ops)

        n_steps = len(buf.obs)
        actions_1hot = util.process_actions(buf.action, self.l_action)
        ones = np.ones(n_steps)
        # Create mask for the current trajectory (all 1s since this is original trajectory)
        mask = np.ones(n_steps)
        
        feed = {
            self.obs: buf.obs,
            self.action_taken: actions_1hot, 
            self.r_ext: buf.reward,
            self.ones: ones,
            self.epsilon: epsilon,
            self.mask: mask,  # Add mask to feed dictionary
            self.action_others: util.get_action_others_1hot_batch(
            buf.action_all, self.agent_id, self.l_action)
        }

        # Process rewards from other agents
        sum_r_from_other = []
        for reward in buf.r_from_others:
            temp = np.sum(reward, axis=0, keepdims=False)
            sum_r_from_other.append(temp[self.agent_id])
            
        if len(sum_r_from_other) < len(buf.reward):
            padding = [0] * (len(buf.reward) - len(sum_r_from_other))
            sum_r_from_other.extend(padding)
            
        feed[self.r_from_others] = sum_r_from_other

        # Compute meta returns using total rewards
        total_rewards = buf.reward + np.array(sum_r_from_other)
        meta_returns = util.process_rewards(total_rewards, self.gamma)
        feed[self.meta_returns] = meta_returns

        # Run both standard and meta updates
        # Make sure all required ops are included
        ops_to_run = [
           self.policy_op_prime,  # Standard policy update
           self.meta_train_op,    # Meta-learning update  
           self.meta_loss         # Meta loss value
        ]
        if self.include_cost_in_chain_rule:
           feed[self.r_given] = buf.r_given

        # Execute all operations
        results = sess.run(ops_to_run, feed_dict=feed)
        meta_loss = results[-1]  # Last item is meta_loss
        return meta_loss

    def train_reward(self, sess, list_buf, list_buf_new, epsilon,
                     summarize=False, writer=None):
        """Train reward function with meta-learning."""
        buf_self = list_buf[self.agent_id]
        buf_self_new = list_buf_new[self.agent_id]

        # Get max length between trajectories
        max_length = max(len(buf_self.obs), len(buf_self_new.obs))
    
        # Create masks for valid timesteps (1 for real, 0 for padded)
        mask_orig = np.ones(max_length)
        mask_orig[len(buf_self.obs):] = 0
        mask_new = np.ones(max_length)
        mask_new[len(buf_self_new.obs):] = 0
    
        # Pad buffers to max length
        def pad_buffer(buf, max_len):
            """Pad buffer arrays to specified length."""
            # Pad observations
            pad_obs = np.zeros([max_len - len(buf.obs)] + list(buf.obs[0].shape))
            buf.obs = np.concatenate([buf.obs, pad_obs], axis=0)
        
            # Pad actions with integers
            pad_act = np.zeros(max_len - len(buf.action), dtype=int)  # Use int type for padding
            buf.action = np.concatenate([buf.action, pad_act])
        
            # Pad rewards
            pad_rew = np.zeros(max_len - len(buf.reward))
            buf.reward = np.concatenate([buf.reward, pad_rew])
        
            # Pad r_from_others (if exists)
            if len(buf.r_from_others) > 0:
                pad_r_others = np.zeros([max_len - len(buf.r_from_others)] + list(buf.r_from_others[0].shape))
                buf.r_from_others = np.concatenate([buf.r_from_others, pad_r_others], axis=0)
        
            # Pad action_all 
            if hasattr(buf, 'action_all') and len(buf.action_all) > 0:
               n_agents = len(buf.action_all[0])
               pad_act_all = [[0] * n_agents for _ in range(max_len - len(buf.action_all))]  # Integer zeros
               buf.action_all.extend(pad_act_all)
        
            # Pad r_given if needed
            if hasattr(buf, 'r_given') and len(buf.r_given) > 0:
                pad_r_given = np.zeros(max_len - len(buf.r_given))
                buf.r_given = np.concatenate([buf.r_given, pad_r_given])
        
            return buf

        # Pad both buffers
        buf_self = pad_buffer(buf_self, max_length)
        buf_self_new = pad_buffer(buf_self_new, max_length)


        ones = np.ones(max_length)
        feed = {}

        # Process rewards for returns calculation
        for agent in self.list_of_agents:
            other_id = agent.agent_id
            if other_id == self.agent_id:
                continue
                
            buf_other = list_buf[other_id]
            buf_other = pad_buffer(buf_other, max_length)
            actions_other_1hot = util.process_actions(
                buf_other.action, self.l_action)
                
            feed[agent.obs] = buf_other.obs
            feed[agent.action_taken] = actions_other_1hot
            feed[agent.r_ext] = buf_other.reward * mask_orig  # Apply mask
            feed[agent.ones] = ones
            feed[agent.epsilon] = epsilon
            feed[agent.action_others] = util.get_action_others_1hot_batch(
                buf_other.action_all, other_id, agent.l_action)

            # Process new buffer
            buf_other_new = list_buf_new[other_id]
            buf_other_new = pad_buffer(buf_other_new, max_length)

            actions_other_1hot_new = util.process_actions(
                buf_other_new.action, self.l_action)
            other_policy_new = self.list_policy_new[other_id]
            feed[other_policy_new.obs] = buf_other_new.obs
            feed[other_policy_new.action_taken] = actions_other_1hot_new

        # process current agent's action_taken specifically
        actions_self_1hot = util.process_actions(buf_self.action, self.l_action)
        feed[self.action_taken] = actions_self_1hot    

        # Calculate returns from new trajectory
        if self.include_cost_in_chain_rule:
            total_reward = [buf_self_new.reward[idx] * mask_new[idx] + 
                       buf_self_new.r_from_others[idx] * mask_new[idx] -
                       buf_self_new.r_given[idx] * mask_new[idx] 
                       for idx in range(max_length)]
        else:
            total_reward = buf_self_new.reward * mask_new
        
        # Calculate discounted returns
        returns = util.process_rewards(total_reward, self.gamma)
    
        # Add returns to feed dict
        feed[self.returns] = returns  
        feed[self.obs] = buf_self.obs
        feed[self.action_others] = util.get_action_others_1hot_batch(
            buf_self.action_all, self.agent_id, self.l_action)
        feed[self.ones] = ones
        feed[self.meta_returns] = returns  # Also feed meta returns

        # Handle meta-network updates for reward recipients
        if not self.can_give:
           # Create meta-network gradient ops if not already created
           if not hasattr(self, 'meta_grads'):
               # Add mask placeholder if not already created
               if not hasattr(self, 'mask'):
                   self.mask = tf.placeholder(tf.float32, [None], 'mask')
                
                # Modify meta loss to use masking
               self.masked_meta_loss = self.meta_loss * tf.reduce_mean(self.mask)
            
               self.meta_grads = tf.gradients(
                    self.masked_meta_loss,
                    self.meta_params
                )
               self.meta_update_op = self.meta_opt.apply_gradients(
                    zip(self.meta_grads, self.meta_params)
                )

            # Add mask to feed dict
           feed[self.mask] = mask_new
           feed[self.meta_policy] = self.probs
        
           ops_to_run = [
               self.reward_op,
               self.meta_update_op,
               self.masked_meta_loss
            ]
        else:
            ops_to_run = [self.reward_op, self.meta_loss]
        
        if self.separate_cost_optimizer:
            ops_to_run.append(self.cost_op)

        # Run all updates
        results = sess.run(ops_to_run, feed_dict=feed)

        return results[1]  # Return meta loss

    def update_main(self, sess):
        """Update main network parameters."""
        sess.run(self.list_copy_prime_to_main_ops)

    def set_can_give(self, can_give):
        self.can_give = can_give    

class PolicyNew(object):
    """Helper class for policy updates."""
    
    def __init__(self, params, l_obs, l_action, agent_name):
        self.obs = tf.placeholder(tf.float32, [None, l_obs], 'obs_new')
        self.action_taken = tf.placeholder(
            tf.float32, [None, l_action], 'action_taken')
            
        prefix = agent_name + '/policy_main/policy/'
        with tf.variable_scope('policy_new'):
            h1 = tf.nn.relu(
                tf.nn.xw_plus_b(
                    self.obs, 
                    params[prefix + 'actor_h1/kernel:0'],
                    params[prefix + 'actor_h1/bias:0']))
            h2 = tf.nn.relu(
                tf.nn.xw_plus_b(
                    h1,
                    params[prefix + 'actor_h2/kernel:0'], 
                    params[prefix + 'actor_h2/bias:0']))
            out = tf.nn.xw_plus_b(
                h2,
                params[prefix + 'actor_out/kernel:0'],
                params[prefix + 'actor_out/bias:0'])
        self.probs = tf.nn.softmax(out)