import os
import numpy as np
import pandas as pd
import webbrowser # TODO: unneccessary
from kaggle_environments import make, evaluate
from training import ConnectFourGym, get_win_percentages
from agent import Agent, lookahead_agent

import torch as th


if __name__ == "__main__":

    env = ConnectFourGym(agent2='random')  
    agent = Agent(env)
    agent.load('default-320000', env, {'learning_rate':0.01, 'batch_size': 128})
    agent.train(timesteps=10000)
    agent.save('default-330000')
    

    def function(obs, conf):
        import random
        col, _ = agent.model.predict(np.array(obs['board']).reshape(6,7,1)) # TODO: Connect-4 specific so far
        #is_valid = (obs['board'][int(col)] == 0)
        #if is_valid:
        return int(col)
        #else:
            #return random.choice([col for col in range(config.columns) if obs.board[int(col)] == 0])

    get_win_percentages('random', function, n_rounds=1000)
    print("Games played: " + str(env.games_played))
    agent.plot()

