from gym.spaces import discrete
import gym
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tilecoding.representation import TileCoding
from ApproximatedSarsaLambdaAgent import ApproximatedSarsaLambdaAgent


class StaticHeuristicApproximatedSarsaLambdaAgent(ApproximatedSarsaLambdaAgent):
    '''
    Agent implementing approximated Sarsa-learning with traces.
    observation_space_mins = an ordered array with minimum values for each of the features of the exploration space
    observation_space_maxs = an ordered array with maximum values for each of the features of the exploation space
    num tiles = how many times do we split a dimension
    num tilings = how many overlapping tilings per dimension.
    
    Replacing traces model, typical replacing traces with tile coding
    TrueOnline Model, True online model!
    
    '''
    
    def __init__(self, observation_space_mins, observation_space_maxs, actions, num_tiles_l, num_tilings_l,  **userconfig):
        super(self.__class__, self).__init__(observation_space_mins, observation_space_maxs, actions, num_tiles_l, num_tilings_l,  **userconfig)
        
        config = {
            "nu": 1, # default value
            "psi": 0.001, # default value
            "static_heuristic": None
            }
        self.config.update(config)
        
        my_config = userconfig.get("my_config")
        self.config.update(my_config)
         
         
        
    def act(self, observation, eps=None):
        
        
        if self.config["static_heuristic"]==None:
            return super.act(self,observation,eps)
        
        else:
        
            heur = self.config["static_heuristic"]
            
            if eps is None:
                eps = self.config["eps"]
            # epsilon greedy.
            preferred_action = heur.get_action(observation)
            possible_actions_values=[] 
            
            for i in range(0,self.action_n):
                possible_actions_values.append(np.sum(self.select_state_action_weights(self.tile_code_weights,observation,i)))
            
            
            chance = np.random.random() 
            if chance> eps:
                action = np.argmax(possible_actions_values)
            else:
                action=self.action_space.sample()
                #action = np.random.randint(0,3)
            return action
        
