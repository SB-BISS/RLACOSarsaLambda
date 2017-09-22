
'''
This is a model of evolution of the environment that the agent can
use to evaluate what would be the next state.

'''
import numpy as np

class maze_model(object):
    pass


    #in the case of the maze it is only about going in one of the four directions
    def next_state(self, current_observation, action):
    
        COMPASS = [ [0, -1],[1, 0], [0, 1], [-1, 0]]
        
        NextState =  current_observation + COMPASS[action]
        return np.array(NextState) # easy