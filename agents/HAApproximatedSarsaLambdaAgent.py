from gym.spaces import discrete
import gym
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tilecoding.representation import TileCoding
from ApproximatedSarsaLambdaAgent import ApproximatedSarsaLambdaAgent
import sys

class HAApproximatedSarsaLambdaAgent(ApproximatedSarsaLambdaAgent):
    '''
    Agent implementing approximated Sarsa-learning with traces and ACO pheromone
    observation_space_mins = an ordered array with minimum values for each of the features of the exploration space
    observation_space_maxs = an ordered array with maximum values for each of the features of the exploation space
    num tiles = how many times do we split a dimension
    num tilings = how many overlapping tilings per dimension.
    
    Replacing traces model, typical replacing traces with tile coding
    TrueOnline Model, True online model
    
    Please notice that in what follows, readability has been preferred to inheritance. So there are long functions
    and code is duplicated to allow people to understand better what the functions do.
    
    
    '''
    
    def __init__(self, observation_space_mins, observation_space_maxs, actions, num_tiles_l, num_tilings_l,  **userconfig):
        #if not isinstance(observation_space, discrete.Discrete):
        #    raise UnsupportedSpace('Observation space {} incompatible with {}. (Only supports Discrete observation spaces.)'.format(observation_space, self))
        #if not isinstance(action_space, discrete.Discrete):
        #    raise UnsupportedSpace('Action space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(action_space, self))
        super(self.__class__, self).__init__(observation_space_mins, observation_space_maxs, actions, num_tiles_l, num_tilings_l,  **userconfig)
        self.config['psi'] = 0.001
        self.config['rho'] = 0.9
        self.config['Pheromone_strategy'] = "hard"
        self.config['nu'] =1
        conf = userconfig.get("my_config")
        self.config.update(conf)
        self.rho = self.config["rho"]
        self.nu = self.config["nu"]
        self.psi = self.config["psi"]
        self.heuristic_activate = False
        self.strategy = self.config["Strategy"]
        print self.strategy
        # a trajectory contains state,action values history., plus the cost to pass back
        self.best_trajectory = {"trajectory" : [], "cost" : sys.maxint, "success" : False} # only succesfull trajectories should be enforced
        self.pheromone_trace = np.zeros((actions.n,self.tile_code_trace_weights[0].size)) +0.1 # this trace will include a pheromone value, initialized negative so that it cannot be chosen by mistake
        self.current_trajectory = {}
        self.current_trajectory_to_ref={}
    #Act overrides the variable    
    def act(self, observation, eps=None):
        if eps is None:
            eps = self.config["eps"]
        # epsilon greedy.
        
        possible_actions_values=[] 
        possible_actions_values_pheromones = []
        
        for i in range(0,self.action_n):
            possible_actions_values.append(np.sum(self.select_state_action_weights(self.tile_code_weights,observation,i)))
            possible_actions_values_pheromones.append(np.sum(self.select_state_action_weights(self.pheromone_trace,observation,i)))
        
        
        #print possible_actions_values
        chance = np.random.random() 
        if chance> eps:
            
            #if there is a success, activate
            if self.heuristic_activate:  
                
                #now select how should the pheromone be considered
                if self.config["Pheromone_strategy"] == "hard":      
                    action = self.hard_heuristic(possible_actions_values,possible_actions_values_pheromones)
                else:
                    action = self.soft_heuristic(possible_actions_values,possible_actions_values_pheromones)
 
            else:
                action = np.argmax(possible_actions_values)
            
        else:
            action=self.action_space.sample()
           
            #action = np.random.randint(0,3)
        return action
    
    
    
    def hard_heuristic(self,possible_actions_values,possible_actions_values_pheromones):
        action = np.argmax(possible_actions_values)
        
        #give me the maximum values in the cell
        action2s = np.argwhere(possible_actions_values_pheromones == possible_actions_values_pheromones[np.argmax(possible_actions_values_pheromones)])
    
        action2 = action
        if len(action2s)==1:
            action2 = np.argmax(possible_actions_values_pheromones)
        else:
            action2 = action2s[np.random.randint(len(action2s))][0]
            
            
        #conservatively, SARSA is preferred if the pheromone level is the same        
        if action!=action2 and possible_actions_values_pheromones[action]!=possible_actions_values_pheromones[action2]: 
            
            val = possible_actions_values[action]
            val2 = possible_actions_values[action2]
            difference_term = self.psi*(val-val2 +self.nu) #this plus 1 is nu
            #print difference_term
            possible_actions_values[action2] = possible_actions_values[action2] + difference_term
                
            actiondel = np.argmax(possible_actions_values)
               
            if actiondel!= action:
                action = actiondel
                #print "Heuristics"
        return action
    
    
    
    def soft_heuristic(self,possible_actions_values,possible_actions_values_pheromones):
        action = np.argmax(possible_actions_values)
        
        probs = possible_actions_values_pheromones/np.sum(possible_actions_values_pheromones)
        
        action2 =0
        val_r = np.random.random()
        
        cumsumprobs = 0
        
        #probabilistic choice
        for z in range(len(probs)):
            cumsumprobs = cumsumprobs + probs[z]
            if val_r <= cumsumprobs:
                action2 = z
                
        #conservatively, SARSA is preferred if the pheromone level is the same        
        if action!=action2 and possible_actions_values_pheromones[action]!=possible_actions_values_pheromones[action2]: 
            val = possible_actions_values[action]
            val2 = possible_actions_values[action2]
            difference_term = self.psi*(val-val2 +self.nu) #this plus 1 is nu
            #print difference_term
            possible_actions_values[action2] = possible_actions_values[action2] + difference_term
                
            actiondel = np.argmax(possible_actions_values)
               
            if actiondel!= action:
                action = actiondel
                #print "Heuristic"
        return action
    
    

    def learn(self, env, rend = False):
        config = self.config
        
        s = env.reset()
        
        #tiles = self.tile_coder
        #self.trace= self.create_state_action_table() # reset
        #trace = self.trace
        #self.tile_coder_traces.reset() # reset replacing traces
        if self.strategy == "Replace":
            self.tile_code_trace_weights = self.tile_code_trace_weights*0
        #traces = self.tile_coder_traces
        
        self.current_trajectory = {}
        present_value_old = 0
        a= self.act(s)
        
        for t in range(config["n_iter"]):
            
            if self.decrease_exploration:
                self.config["eps"] = self.config["eps"]*0.99
                    
            
            if self.strategy=="Replacing":
               
                dict = {"state":s,"action":a}
                
                self.current_trajectory[str(s*1.0)+"_"+str(a*1.0)] = dict #since it is an hashmap, now it will not store the state more than once
                
                sp, reward, done, _ = env.step(a)
                            
                ap= self.act(sp)
                
                index_present = self.tile_coder[a].__call__(s)
                index_future = self.tile_coder[ap].__call__(sp)
                
                future_value = np.sum(self.select_state_action_weights(self.tile_code_weights, sp, ap))    
                present_value = np.sum(self.select_state_action_weights(self.tile_code_weights, s, a)) 
                
                if done:
                    future_value = 0.0
                                
                delta = reward + self.gamma*future_value- present_value
                              
                self.tile_code_trace_weights[a,index_present] = 1
    
                self.tile_code_weights[a,index_present] = self.tile_code_weights[a,index_present] + self.lmbd*delta*self.alpha*self.tile_code_trace_weights[a,index_present] 
                self.tile_code_trace_weights= self.tile_code_trace_weights*self.gamma*self.lmbd
                
                a=ap
                s = sp
                
            if self.strategy=="TrueOnline":  
                                
                dict= {"state":s,"action": a}
                self.current_trajectory[str(s*1.0)+"_"+str(a*1.0)] = dict
                
                sp, reward, done, _ = env.step(a)
                
                ap= self.act(sp)
                
                index_present = self.tile_coder[a].__call__(s)
                index_future = self.tile_coder[ap].__call__(sp)
                 
                #same indices, different weights
                future_value = np.sum(self.select_state_action_weights(self.tile_code_weights, sp, ap))    
                present_value = np.sum(self.select_state_action_weights(self.tile_code_weights, s, a)) 
                
                if done:
                    future_value = 0.0
                
                dutch = (1 -self.alpha*self.gamma*self.lmbd*np.sum(self.tile_code_trace_weights[a,index_present])) # just a value.
                self.tile_code_trace_weights[a,index_present] += dutch
               
                delta = reward + self.gamma*future_value- present_value
                delta_q = (present_value - present_value_old)
                
                #self.tile_code_weights +=  self.alpha * (delta + present_value - present_value_old) * self.tile_code_trace_weights
                self.tile_code_weights +=  self.alpha * (delta) * self.tile_code_trace_weights  +  self.alpha*(delta_q)*self.tile_code_trace_weights 
                self.tile_code_weights[a,index_present] = self.tile_code_weights[a,index_present]- self.alpha *(delta_q)
                
                self.tile_code_trace_weights= self.tile_code_trace_weights*self.gamma*self.lmbd 
                  
                present_value_old = future_value
                  
                a= ap
                s = sp
                
            
            if done: # something wrong in MC
                self.last_steps.append(t)
                
                #if it is succesfful.
                result = self.success_invariant(t=t,env=env)
                
                if result:
                    self.handle_trajectory(result = result, t=t, trajectory= self.current_trajectory)
                    #reinforce if there is a successful trajectory or do nothing
                    
                #print s
                if self.heuristic_activate:
                    self.reinforce_heuristic_model()
                
                #done, exit
                break
            
            if rend:
                env.render()
                
    
    #to be overridden according to the scenario, if no overriding needed just don't
    #change it in the subclass
    #in this case I consider a trajectory succesfull if
    def success_invariant(self,**args):
        t= args.get("t")
        env = args.get("env")
                 
        if t<env._max_episode_steps-1:
            self.heuristic_activate = True
           
            return True
        else:
            self.heuristic_activate = False
            return False
    
    
                
        
    def handle_trajectory(self, **args):
        t= args.get("t")
        result= args.get("result")
        Trajectory = args.get("trajectory")
        if result:
            
            self.current_trajectory_to_ref= {"trajectory" : Trajectory, "cost" : 1.0/t, "success" : True} #
    
            if 1.0/t < self.best_trajectory["cost"]:
                #print 1.0/t
                self.best_trajectory= {"trajectory" : Trajectory, "cost" : 1.0/t, "success" : True} #
                #self.reinforce_heuristic_model()#Don't forget to reinforce in other models a heuristic of an empty vector is never going to work
                # it could be different here
                #one could think about storying multiple trajectories, rank them and enforce them... or else.
                
    
    def reinforce_heuristic_model(self):
        cost = self.best_trajectory["cost"]
        Trajectory = self.best_trajectory["trajectory"]
        for i in Trajectory.keys():
           
           dict = Trajectory[i] # this is a dict
           s= dict["state"] # get the state
           a= dict["action"]# get the action
           #pheromone update rule
           
           index_pheromone_tiles = self.tile_coder[a].__call__(s)
           self.pheromone_trace[a,index_pheromone_tiles] = self.pheromone_trace[a,index_pheromone_tiles] + cost
           #print self.pheromone_trace[a,index_pheromone_tiles]
        cost2 = self.current_trajectory_to_ref["cost"]
        Trajectory = self.current_trajectory_to_ref["trajectory"]
        
        if cost2 != cost:
            for i in Trajectory.keys():
               
               dict = Trajectory[i] # this is a dict
               s= dict["state"] # get the state
               a= dict["action"]# get the action
               #pheromone update rule
               
               index_pheromone_tiles = self.tile_coder[a].__call__(s)
               self.pheromone_trace[a,index_pheromone_tiles] = self.pheromone_trace[a,index_pheromone_tiles] + cost2
              
               
           
        #print "Reinforcing"   
        self.pheromone_trace = self.rho*self.pheromone_trace
                        

    