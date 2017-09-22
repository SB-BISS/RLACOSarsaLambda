import numpy as np
import sys
import scipy

'''
@author Stefano Bromuri

It selects the state that has the minimum distance from the goal.
Static Heuristic for Maze environments

'''

class M3DHeuristic(object):
      
    def __init__(self, model=None, actions_number=0):
        self.model = model#we need a model for the evolution of the environment
        self.action_n = actions_number

    def get_action(self,observation):
        
        valvelocity_x = -10 #velocity x
        valvelocity_y = -10 #velocity y
        action_sel = 0
        for actionf in range(self.action_n):
            S=self.model.next_state(observation,actionf)
            if np.abs(S[1])> np.abs(valvelocity_x) and np.abs(S[3])> np.abs(valvelocity_y) :
                valvelocity_x = np.abs(S[1])
                valvelocity_y = np.abs(S[3])
                action_sel=actionf
                
        return action_sel
    
    
    
