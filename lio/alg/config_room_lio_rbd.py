from lio.utils import configdict


def get_config():

    config = configdict.ConfigDict()

    config.alg = configdict.ConfigDict()
    config.alg.n_episodes = 25000
    config.alg.n_eval = 10
    config.alg.n_test = 100
    config.alg.name = 'lio-rbd'
    config.alg.period = 500

    config.env = configdict.ConfigDict()
    config.env.max_steps = 5
    config.env.min_at_lever = 1
    config.env.n_agents = 2
    config.env.name = 'er'
    config.env.r_multiplier = 2.0
    config.env.randomize = False
    config.env.reward_sanity_check = False

    config.lio = configdict.ConfigDict()
    config.lio.asymmetric = False
    config.lio.decentralized = False
    config.lio.entropy_coeff = 0.01
    config.lio.epsilon_div = 1000
    config.lio.epsilon_end = 0.1
    config.lio.epsilon_start = 0.5
    config.lio.gamma = 0.99
    config.lio.include_cost_in_chain_rule = False
    config.lio.lr_actor = 1e-4
    config.lio.lr_cost = 1e-4
    config.lio.lr_opp = 1e-3
    config.lio.lr_reward = 1e-3
    config.lio.sigma_lr = 1e-3  # Learning rate for sigma parameters
    config.lio.energy_weight = 0.01  # Beta coefficient for energy cost
    config.lio.initial_sigma1 = 1.0  # Initial value for sigma1
    config.lio.initial_sigma2 = 1.0  # Initial value for sigma2
    config.lio.lr_v = 1e-2
    config.lio.optimizer = 'adam'
    config.lio.reg = 'l1'
    config.lio.reg_coeff = 1.0
    config.lio.separate_cost_optimizer = True
    config.lio.tau = 0.01
    config.lio.use_actor_critic = False

    config.main = configdict.ConfigDict()
    config.main.dir_name = 'er_lio_rbd_4_2'
    config.main.exp_name = 'er'
    config.main.max_to_keep = 100
    config.main.model_name = 'model.ckpt'
    config.main.save_period = 100000
    config.main.seed = 12340
    config.main.summarize = False
    config.main.use_gpu = False

    config.nn = configdict.ConfigDict()
    config.nn.n_h1 = 64
    config.nn.n_h2 = 32
    config.nn.n_hr1 = 64
    config.nn.n_hr2 = 16

    return config
                            
                
