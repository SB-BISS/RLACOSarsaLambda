import numpy as np
import math



'''

@author Stefano Bromuri
This is a class whose only function is to provide the next_state for the mountain 3D car problem
basically, this is a model that the agent can use to query what will happen is an action is performed.

In order to be able to use these models with the HAAApprximatedSarsaLambdaAgent 
you only need to implement the next_state method.

For the purpose of tihs library, I am not that interested in knowing the next reward
eventually this model could be extended a bit

 
'''

class m3d_model(object):
    
    def __init__(self):
        self.min_position_x = -1.2
        self.max_position_x = 0.6
        self.max_speed_x = 0.07
        self.goal_position_x = 0.5
        
        self.min_position_y = -1.2
        self.max_position_y = 0.6
        self.max_speed_y = 0.07
        self.goal_position_y = 0.5



    def next_state(self, current_observation, action):
       
        position_x, velocity_x,position_y, velocity_y = current_observation
        
        if action == 4:
            velocity_x += math.cos(3*position_x)*(-0.0025)
            velocity_y += math.cos(3*position_y)*(-0.0025)
                    
        if action == 3:
            velocity_x += 0.002 + math.cos(3*position_x)*(-0.0025)
        if action == 2:
            velocity_y += 0.002 + math.cos(3*position_y)*(-0.0025)
        if action == 1:
            velocity_x += -0.002 + math.cos(3*position_x)*(-0.0025)
        if action == 0:
            velocity_y += -0.002 + math.cos(3*position_y)*(-0.0025)
        
        #typical limits of mountain car, but in two dimensions
        velocity_x = np.clip(velocity_x, -self.max_speed_x, self.max_speed_x)    
        velocity_y = np.clip(velocity_y, -self.max_speed_y, self.max_speed_y)

        position_x += velocity_x
        position_y += velocity_y
        
        position_x = np.clip(position_x, self.min_position_x, self.max_position_x)
        
        position_y = np.clip(position_y, self.min_position_y, self.max_position_y)
        
        if (position_x==self.min_position_x and velocity_x<0): velocity_x = 0
        
        if (position_y==self.min_position_y and velocity_y<0): velocity_y = 0


        next_state = (position_x, velocity_x,position_y,velocity_y)
        return np.array(next_state)
