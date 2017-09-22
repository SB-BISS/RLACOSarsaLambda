import pickle
import numpy as np

'''
@author: Stefano Bromuri

This Heuristic function loads a pheromone trail learned with AP-HARL to be
used in an environment as a static heuristic

Filename is assumed to be a pheromone trace

'''



class LoadableHeuristic(object):
    
    
    def __init__(self,FileName,FileNameTC, actions_number=0):    
        self.pheromone_trace = pickle.load( open( FileName, "rb" ) )
        self.tc = pickle.load( open( FileNameTC, "rb" ) )
        self.action_n = actions_number
    
    def get_action(self,observation):
        
        pher_values = np.zeros(self.action_n)
        for i in range(self.action_n):
           index_p= self.tc[i].__call__(observation)
           pher_values[i] = np.sum(self.pheromone_trace[i][index_p])
           
        return np.argmax(pher_values)
    



