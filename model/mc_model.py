import numpy as np
import math



'''
@author Stefano Bromuri
This is a class whose only function is to provide the next_state for the mountain car problem
basically, this is a model that the agent can use to query what will happen is an action is performed.
 
'''

class mc_model(object):
    pass


    #model based learning, we want to know the next state
    def next_state(self, current_observation, action):
            
            position, velocity = current_observation
            velocity += (action-1)*0.001 + math.cos(3*position)*(-0.0025)
            velocity = np.clip(velocity, -0.07, 0.07)
            position += velocity
            position = np.clip(position, -1.2,0.6)
            
            if (position==-1.2 and velocity<0): velocity = 0

            next_state = np.array([position, velocity])
            return next_state
        
        
        
