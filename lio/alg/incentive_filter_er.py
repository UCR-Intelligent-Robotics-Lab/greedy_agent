"""
Implementation of LIO Filter to defend against exploitation.
This adds a statistical filter that detects and adjusts abnormal incentive patterns.
"""
import numpy as np
import os
import pandas as pd

class IncentiveFilter:
    """
    Filter for detecting and correcting abnormal incentives based on historical data.
    Uses statistical z-score to identify outliers in agent incentive patterns.
    """
    def __init__(self, delta=1, range_size=500):
        """
        Initialize the filter with historical data.
        
        Args:
            delta: Threshold for z-score filtering (default: 1 for 68% confidence)
            range_size: Number of episodes to group into a single statistical bucket
        """
        self.delta = delta
        self.is_initialized = False
        self.policy_means = None
        self.policy_stds = None
        self.incentive_means = None
        self.incentive_stds = None
        self.range_size = range_size
        self.range_stats = {}  # Will store stats for each episodic range
        self.exploit_detection = {}  # Track suspected exploitative agents
        
        
    def load_historical_data(self, max_experiments=10):
        """
        Load and process historical data from previous experiments.
        
        Args:
            max_experiments: Maximum number of experiments to consider
        """

        # Get the base directory for this project
        # If running from train_lio_filter_explotitive_attack_er.py
        # This should be the directory containing that file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Lists to store dataframes from each experiment
        df_all_policy = []
        df_all_incentive = []
        
        files_found = False
        print(f"Looking for CSV files in base directory: {base_dir}")

        for i in range(1, max_experiments + 1):
           # Construct absolute paths
           policy_path = os.path.join(base_dir, f'results/er{i}/er_lio_4_2/given_incentives_sum_policy_log.csv')
           incentive_path = os.path.join(base_dir, f'results/er{i}/er_lio_4_2/given_incentives_sum_incentive_log.csv')
        
           print(f"Checking for files at:\n  - {policy_path}\n  - {incentive_path}")
        
            # Check if both files exist
           if os.path.exists(policy_path) and os.path.exists(incentive_path):
              try:
                   df_policy = pd.read_csv(policy_path)
                   df_incentive = pd.read_csv(incentive_path)
                
                   # Add to lists
                   df_all_policy.append(df_policy)
                   df_all_incentive.append(df_incentive)
                   files_found = True
                   print(f"Successfully loaded data for experiment {i}")
              except Exception as e:
                print(f"Error loading experiment {i}: {e}")
           else:
               print(f"Experiment {i}: Files not found")
    
        # If no files found, return False to indicate initialization failed
        if not files_found:
           print("No historical data found. Running with default parameters.")
           return False


        
            
        # Process the data to get mean and std for each agent at each episode
        max_episodes = 25000
        num_agents = 4
        
        print(f"Max episodes: {max_episodes}, Number of agents: {num_agents}")
        
        # Initialize storage for means and stds
        self.policy_means = np.zeros((num_agents, max_episodes + 1))
        self.policy_stds = np.zeros((num_agents, max_episodes + 1))
        self.incentive_means = np.zeros((num_agents, max_episodes + 1))
        self.incentive_stds = np.zeros((num_agents, max_episodes + 1))
        
      
        # We process data in batches of 500 episodes
        # Process data in episode ranges (only 50 ranges × 4 agents = 200 iterations)
        for range_start in range(1, max_episodes+1, self.range_size):
           range_end = min(range_start + self.range_size - 1, max_episodes)
            # For each agent in this range
           for agent_id in range(num_agents):
        
            # Collect all values in this range across experiments
            
               policy_col = f'agent_{agent_id}_incentives'
               incentive_col = f'agent_{agent_id}_incentives'
               policy_values = []
               incentive_values = []
            
               # Process policy data
               for df in df_all_policy:
                   if policy_col in df.columns:
                   # Filter by episode range
                    range_data = df[(df['episode'] >= range_start) & 
                               (df['episode'] <= range_end)]
                    if not range_data.empty:
                       policy_values.extend(range_data[policy_col].values)
        
                # Process incentive data  
               for df in df_all_incentive:
                   if incentive_col in df.columns:
                       range_data = df[(df['episode'] >= range_start) & 
                               (df['episode'] <= range_end)]
                   if not range_data.empty:
                        incentive_values.extend(range_data[incentive_col].values)
               
          
                # Calculate mean and std for the entire range
                # Only calculate if we have enough data
               if len(policy_values) > 1:
                  mean_val = np.mean(policy_values)
                  std_val = np.std(policy_values)
                  print(f"Range {range_start}-{range_end}, in policy training phase, Agent {agent_id}: given incentive sum mean={mean_val:.4f}, std={std_val:.4f}")
                  # Assign to all episodes in range
                  for episode in range(range_start, range_end+1):
                    self.policy_means[agent_id, episode] = mean_val
                    self.policy_stds[agent_id, episode] = std_val
                
                # Same for incentive values
               if len(incentive_values) > 1:
                    mean_val = np.mean(incentive_values)
                    std_val = np.std(incentive_values)
                    print(f"Range {range_start}-{range_end}, in incentive training phase, Agent {agent_id}: given incentive sum mean={mean_val:.4f}, std={std_val:.4f}")
                    # Assign to all episodes in range
                    for episode in range(range_start, range_end+1):
                        self.incentive_means[agent_id, episode] = mean_val
                        self.incentive_stds[agent_id, episode] = std_val



        
        
        self.is_initialized = True
        print("Filter initialized successfully")
        return True
    
    def calculate_correction(self, agent_id, episode, incentive_sum, is_policy=True):
        """
        Calculate the correction parameter for an agent's incentives.
        
        Args:
            agent_id: The agent ID
            episode: Current episode number
            incentive_sum: Total incentives given by the agent
            is_policy: Whether this is from policy training (True) or incentive training (False)
            
        Returns:
            correction_factor: Value between 0 and 1 to scale incentives
        """
        if not self.is_initialized:
            return 1.0
        
        # Use appropriate means and stds based on training phase
        if is_policy:
            means = self.policy_means
            stds = self.policy_stds
        else:
            means = self.incentive_means
            stds = self.incentive_stds
        
        # If episode number is beyond our historical data, use the last available
        if episode >= means.shape[1]:
            episode = means.shape[1] - 1
        
        # If std is zero or very small, don't filter
        if stds[agent_id, episode] < 1e-6:
            print(f"Agent {agent_id} has zero std (z=0). No correction needed.")
            return 1.0
        
        # Calculate z-score
        z_score = abs(incentive_sum - means[agent_id, episode]) / stds[agent_id, episode]
        
        # Determine correction factor
        if z_score <= self.delta:
            # Within normal range, no correction needed
            print(f"Agent {agent_id} within normal range (z={z_score:.2f}). No correction needed.")
            return 1.0
        elif means[agent_id, episode] < incentive_sum:
            # Incentives are abnormally high, scale them down
            correction = min(1.0, means[agent_id, episode] / incentive_sum)
            print(f"Agent {agent_id} giving abnormally high incentives (z={z_score:.2f}). Correction: {correction:.4f}")
            return correction
        else:
            # Incentives are abnormally low, scale them up
            correction = max(1.0, means[agent_id, episode] / incentive_sum)
            print(f"Agent {agent_id} giving abnormally low incentives (z={z_score:.2f}). Correction: {correction:.4f}")
            return correction