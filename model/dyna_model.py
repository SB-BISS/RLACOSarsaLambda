
'''
This is a model of evolution of the environment that the agent can
use to evaluate what would be the next state.

'''
import numpy as np

class dyna_model(object):
    
    def __init__(self,width = 30, height=30):
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]
        self.WORLD_WIDTH = width
        # maze height
        self.WORLD_HEIGHT = height

    
    
    def next_state(self, state, action):
        x, y = state

        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.WORLD_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.WORLD_WIDTH - 1)
        #the trajectory to draw     
        #print [x,y]  
        return np.array([x, y])