import os
import numpy as np
import pandas as pd
import webbrowser # TODO: unneccessary
from kaggle_environments import make, evaluate
from training import ConnectFourGym, get_win_percentages
from agent import Agent, lookahead_agent
from stable_baselines3.common.env_checker import check_env

import torch as th


if __name__ == "__main__":

    env = ConnectFourGym(agent2=lookahead_agent)
    #check_env(env, warn=True, skip_render_check=True)
    agent = Agent(env)
    policy_dir = 'ppo_cnn/'
    #agent.load(policy_dir + 'ppo_cnn-250000', env, {'learning_rate':0.0002, 'device':'cpu'})
    agent.train(timesteps=1000)
    #agent.save(policy_dir + 'ppo_cnn-251000')
    

    def function(obs, conf):
        import random
        col, _ = agent.model.predict(np.array(obs['board']).reshape(1,6,7)) # TODO: Connect-4 specific so far
        #is_valid = (obs['board'][int(col)] == 0)
        #if is_valid:
        return int(col)
        #else:
            #return random.choice([col for col in range(config.columns) if obs.board[int(col)] == 0])

    get_win_percentages(lookahead_agent, function, n_rounds=100)
    agent.plot()

