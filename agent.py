import os, gym, datetime
import torch as th
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Neural network for predicting action values
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

class CustomCnnPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomCnnPolicy, self).__init__(*args, **kwargs, normalize_images = False)

class Agent(object):

    def __init__(self, env, model=None):
        if model:
            self.model = model
        else:
            self.log_dir = "ppo_cnn/" + str(datetime.datetime.now()).replace(":", "-")
            os.makedirs(self.log_dir, exist_ok=True)
            monitor_env = Monitor(env, self.log_dir, allow_early_resets=True)
            vec_env = DummyVecEnv([lambda: monitor_env])
            policy_kwargs = dict(features_extractor_class=CustomCNN,
                                    features_extractor_kwargs=dict(features_dim=256), net_arch=[dict(pi=[64, 64], vf=[64, 64])])
            self.model=PPO(CustomCnnPolicy, vec_env, policy_kwargs=policy_kwargs, verbose=1, learning_rate = 0.001)

    def function(self, obs, conf):
        import random
        col, _ = self.model.predict(np.array(obs['board']).reshape(6,7,1)) # TODO: Connect-4 specific so far
        is_valid = (obs['board'][int(col)] == 0)
        if is_valid:
            return int(col)
        else:
            return random.choice([col for col in range(config.columns) if obs.board[int(col)] == 0])

    def train(self, timesteps):
        self.model.learn(total_timesteps=timesteps)

    def save(self, name : str):
        self.model.save(name)

    def load(self, name : str, env, replace_parameters = None):
        self.log_dir = "ppo_cnn/" + str(datetime.datetime.now()).replace(":", "-")
        os.makedirs(self.log_dir, exist_ok=True)
        monitor_env = Monitor(env, self.log_dir, allow_early_resets=True)
        vec_env = DummyVecEnv([lambda: monitor_env])
        self.model = PPO.load(name, env=vec_env, custom_objects = replace_parameters)

    def plot(self):
             # Plot cumulative reward
        with open(os.path.join(self.log_dir, "monitor.csv"), 'rt') as fh:    
            firstline = fh.readline()
            assert firstline[0] == '#'
            df = pd.read_csv(fh, index_col=None)['r']
        df.rolling(window=1000).mean().plot()
        plt.show()

def lookahead_agent(obs, config):
    import random
    import numpy as np
    # Calculates score if agent drops piece in selected column
    def score_move(grid, col, mark, config):
        next_grid = drop_piece(grid, col, mark, config)
        score = get_heuristic(next_grid, mark, config)
        return score

    # Helper function for score_move: gets board at next step if agent drops piece in selected column
    def drop_piece(grid, col, mark, config):
        next_grid = grid.copy()
        for row in range(config.rows-1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = mark
        return next_grid

    # Helper function for score_move: calculates value of heuristic for grid
    def get_heuristic(grid, mark, config):
        num_threes = count_windows(grid, 3, mark, config)
        num_fours = count_windows(grid, 4, mark, config)
        num_threes_opp = count_windows(grid, 3, mark%2+1, config)
        num_fours_opp = 0
        for col in range(config.columns):
            num_fours_opp += count_windows(drop_piece(grid, col, mark%2+1, config), 4, mark%2+1, config)
        score = num_threes - 1e2*num_threes_opp + 1e6*num_fours - 1e5*num_fours_opp
        return score

    # Helper function for get_heuristic: checks if window satisfies heuristic conditions
    def check_window(window, num_discs, piece, config):
        return (window.count(piece) == num_discs and window.count(0) == config.inarow-num_discs)

    # Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
    def count_windows(grid, num_discs, piece, config):
        num_windows = 0
        # horizontal
        for row in range(config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[row, col:col+config.inarow])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # vertical
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns):
                window = list(grid[row:row+config.inarow, col])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # positive diagonal
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # negative diagonal
        for row in range(config.inarow-1, config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        return num_windows
    
    # Get list of valid moves
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    # Use the heuristic to assign a score to each possible board in the next turn
    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config) for col in valid_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return random.choice(max_cols)

