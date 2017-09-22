import numpy as np
import sys
import scipy

'''
@author Stefano Bromuri

It selects the state that has the minimum distance from the goal.
Static Heuristic for Maze environments

'''

class MountainCarHeuristic(object):
      
    def __init__(self, model=None, actions_number=3):
        self.model = model#we need a model for the evolution of the environment
        self.action_n = actions_number

    def get_action(self,observation):
        
        valvelocity = -10 #velocity
        action_sel = 0
        for actionf in range(self.action_n):
            S=self.model.next_state(observation,actionf)
            if np.abs(S[1])> np.abs(valvelocity):
                valvelocity = np.abs(S[1])
                action_sel=actionf
                
        return action_sel
    
    
    

