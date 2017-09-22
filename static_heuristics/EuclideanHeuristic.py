import numpy as np
import sys
import scipy

'''
@author Stefano Bromuri

It selects the state that has the minimum distance from the goal.
Static Heuristic for Maze environments

'''

class EuclideanHeuristic(object):
      
    def __init__(self,goal=None, model=None, actions_number=0):
        self.goal = goal
        self.model = model#we need a model for the evolution of the environment
        self.action_n = actions_number

    def get_action(self,observation):
        
        dist = sys.float_info.max
        action_sel = 0
        for actionf in range(self.action_n):
            S=self.model.next_state(observation,actionf)
            cdist = np.linalg.norm(S-self.goal)
            if cdist< dist:
                dist = cdist
                action_sel=actionf
                
        return action_sel