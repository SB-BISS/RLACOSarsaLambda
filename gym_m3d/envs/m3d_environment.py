

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class m3d(gym.Env):
    
    def __init__(self):
        self.min_position_x = -1.2
        self.max_position_x = 0.6
        self.max_speed_x = 0.07
        self.goal_position_x = 0.5
        
        self.min_position_y = -1.2
        self.max_position_y = 0.6
        self.max_speed_y = 0.07
        self.goal_position_y = 0.5
        
        print "initiating m3d..."

        self.low = np.array([self.min_position_x, -self.max_speed_x, self.min_position_y, -self.max_speed_y])
        self.high = np.array([self.max_position_x, self.max_speed_x, self.max_position_y, self.max_speed_y])

        self.viewer = None

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(self.low, self.high)

        self._seed()
        self.reset()
        print "initiated m3d..."

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position_x, velocity_x,position_y, velocity_y = self.state
        
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

        done = bool(position_x >= self.goal_position_x and position_y >= self.goal_position_y)
        reward = -1.0

        self.state = (position_x, velocity_x,position_y,velocity_y)
        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0,self.np_random.uniform(low=-0.6, high=-0.4),0])
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    
    def _render(self, mode='human', close=False):
        raise Exception("Not supported.")
        