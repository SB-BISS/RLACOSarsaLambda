from gym.spaces import discrete
import gym
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tilecoding.representation import TileCoding
from ApproximatedSarsaLambdaAgent import ApproximatedSarsaLambdaAgent
import sys
import time
import pickle


class HAApproximatedSarsaLambdaAgent(ApproximatedSarsaLambdaAgent):
    '''
    @author Stefano Bromuri
    
    Agent implementing approximated Sarsa-learning with traces and ACO pheromone given a model of the environment.
    
    You can see this as a pheromone based model learning.
    
    observation_space_mins = an ordered array with minimum values for each of the features of the exploration space
    observation_space_maxs = an ordered array with maximum values for each of the features of the exploation space
    num tiles = how many times do we split a dimension
    num tilings = how many overlapping tilings per dimension.
    
    Replacing traces model, typical replacing traces with tile coding
    TrueOnline Model, True online model
    
    Please notice that in what follows, readability has been preferred to inheritance. So there are long functions
    and code is duplicated to allow people to understand better what the functions do.
    
    For the parameters, look in the superclass for the RL part. 
    
    in the config one should add the following parameters:
    
    psi: importance of the heuristic part,  a number in [0,1]
    nu: bias on the heuristic part. An integer (usually 1,5 or 10). 
    model_based: boolean, should we use a model of the environment or not
    model: the model to be used in the forward action selection
    rho: pheromone discount factor.
    
    '''
    
    def __init__(self, observation_space_mins, observation_space_maxs, actions, num_tiles_l, num_tilings_l,  **userconfig):
        #if not isinstance(observation_space, discrete.Discrete):
        #    raise UnsupportedSpace('Observation space {} incompatible with {}. (Only supports Discrete observation spaces.)'.format(observation_space, self))
        #if not isinstance(action_space, discrete.Discrete):
        #    raise UnsupportedSpace('Action space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(action_space, self))
        super(self.__class__, self).__init__(observation_space_mins, observation_space_maxs, actions, num_tiles_l, num_tilings_l,  **userconfig)
        self.config['psi'] = 0.001
        self.config['rho'] = 0.9
        self.config["model_based"]=False
        self.config["model"] = None
        self.config['Pheromone_strategy'] = "hard"
        self.config['nu'] =1
        self.config["descrease_exploration_rate"] = 0.99
        conf = userconfig.get("my_config")
        self.config.update(conf)
        self.model = self.config["model"]
        self.model_based= self.config["model_based"]
        self.rho = self.config["rho"]
        self.decrease_exploration_rate = self.config["descrease_exploration_rate"]
        self.nu = self.config["nu"]
        self.psi = self.config["psi"]
        self.heuristic_activate = False
        self.strategy = self.config["Strategy"]
        print self.strategy    
        # a trajectory contains state,action values history., plus the cost to pass back
        self.best_trajectory = {"trajectory" : [], "quality" : 0, "success" : False} # only succesfull trajectories should be enforced
        self.pheromone_trace = np.zeros((actions.n,self.tile_code_trace_weights[0].size)) +0.1 # this trace will include a pheromone value, initialized negative so that it cannot be chosen by mistake
        self.current_trajectory = {}
        self.current_trajectory_to_ref={}
        self.hard = True
        
        if self.config['Pheromone_strategy'] == "hard":
            self.hard=True
        else:
            self.hard=False
        
        
    #Act overrides the variable  
      
    def act(self, observation, eps=None):
        if eps is None:
            eps = self.config["eps"]
        # epsilon greedy.
        
        #previous_pheromone_value =
        
        
        possible_actions_values=np.zeros(self.action_n) 
        possible_actions_values_pheromones = np.zeros(self.action_n)
        
        for i in range(0,self.action_n):
            possible_actions_values[i]=(np.sum(self.select_state_action_weights(self.tile_code_weights,observation,i)))
            possible_actions_values_pheromones[i]=(np.sum(self.select_state_action_weights(self.pheromone_trace,observation,i)))
        
        
        #print possible_actions_values
        chance = np.random.random() 
        if chance> eps:
            
            #if there is a success, activate
            if self.heuristic_activate:  
                
                #now select how should the pheromone be considered
                if self.hard:      
                    action = self.hard_heuristic(possible_actions_values,possible_actions_values_pheromones)
                else:
                    action = self.soft_heuristic(possible_actions_values,possible_actions_values_pheromones)
 
            else:
                action = np.argmax(possible_actions_values)
            
        else:
            action=self.action_space.sample()
           
            #action = np.random.randint(0,3)
        return action
    
    
    #Model based action.
    
    def act2(self, observation, eps=None):
        if eps is None:
            eps = self.config["eps"]
        # epsilon greedy.
        
        #previous_pheromone_value =
        
        #previous_values_pheromones = []
        
        observations = []
        possible_actions_values_pheromones = []
        possible_actions_values=[]  
        
        for actionf in range(self.action_n):
            possible_actions_values.append(np.sum(self.select_state_action_weights(self.tile_code_weights,observation,actionf)))
            f_obs = self.model.next_state(observation,actionf) #so this is my next state
            
            pheromone_present= np.sum(self.select_state_action_weights(self.pheromone_trace,observation,actionf))
            pheromone_future_transition=[]
            
            for f_act in range(self.action_n): # I sum all the pheromone in the next state !
                    pheromone_future_transition.append(np.sum(self.select_state_action_weights(self.pheromone_trace,f_obs,f_act)))
            possible_actions_values_pheromones.append(np.max(pheromone_future_transition)+pheromone_present) # as a measure of the variability        
              
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
        
        #remember that if action has already the highest pheromon it simply is the best action also for ACO.
        action = np.argmax(possible_actions_values)
        #give me the maximum values in the cell
        action2 = np.argmax(possible_actions_values_pheromones)
                              
        #conservatively, SARSA is preferred if the pheromone level is the same        
        if action!=action2: #and possible_actions_values_pheromones[action]!=possible_actions_values_pheromones[action2]: 
            
            val = possible_actions_values[action]
            val2 = possible_actions_values[action2]
            difference_term = self.psi*(val-val2 +self.nu) #this plus 1 is nu
            #print difference_term
            val2 = val2 + difference_term
                
            if val2>val:
                action = action2
                #print time.time()
        return action
    
    
    
    def soft_heuristic(self,possible_actions_values,possible_actions_values_pheromones):
        action = np.argmax(possible_actions_values)
        
        bias = possible_actions_values/np.sum(possible_actions_values)
        
        probs = np.exp(possible_actions_values_pheromones)/np.sum(np.exp(possible_actions_values_pheromones))
        
        #renormalize
        
        probs = probs/np.sum(probs)
        
        action2 =0
        val_r = np.random.random()
        
        cumsumprobs = 0
        
        #probabilistic choice
        for z in range(len(probs)):
            cumsumprobs = cumsumprobs + probs[z]
            if val_r <= cumsumprobs:
                action2 = z
                
        #conservatively, SARSA is preferred if the pheromone level is the same        
        if action!=action2:# and possible_actions_values_pheromones[action]!=possible_actions_values_pheromones[action2]: 
            val = possible_actions_values[action]
            val2 = possible_actions_values[action2]
            difference_term = self.psi*(val-val2 +self.nu) #this plus 1 is nu
            #print difference_term
            possible_actions_values[action2] = possible_actions_values[action2] + difference_term
                
            actiondel = np.argmax(possible_actions_values)
               
            if actiondel!= action:
                action = actiondel
                #print time.time()
                #print probs

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
        aold = a
        sold = s
        self.cum_reward = 0
        for t in range(config["n_iter"]):
            
            if self.decrease_exploration:
                self.config["eps"] = self.config["eps"]*self.decrease_exploration_rate
                    
            
            if self.strategy=="Replacing":
               
                dict = {"state":s,"action":a, "time":t}#and when
                
                self.current_trajectory[str(s*1.0)+"_"+str(a*1.0)] = dict #since it is an hashmap, now it will not store the state more than once
                
                sp, reward, done, _ = env.step(a)
                self.cum_reward = self.cum_reward + reward
                
                if self.model_based:      
                    ap= self.act2(sp)
                else:
                    ap = self.act(sp)
                
                
                index_present = self.tile_coder[a].__call__(s)
                index_future = self.tile_coder[ap].__call__(sp)
                
                future_value = np.sum(self.select_state_action_weights(self.tile_code_weights, sp, ap))    
                present_value = np.sum(self.select_state_action_weights(self.tile_code_weights, s, a)) 
                
                #Pheromone part
                #move the pheromone away towards good trajectories, and discount it
                #the idea of this part really is to make the pheromone shape variable in the environment

                pher_future_value = np.sum(self.select_state_action_weights(self.pheromone_trace, sp, ap))    
                pher_present_value = np.sum(self.select_state_action_weights(self.pheromone_trace, s, a)) 
              
                if np.sum(self.pheromone_trace[ap,index_future])<=np.sum(self.pheromone_trace[a,index_present]):
                    self.pheromone_trace[ap,index_future] = self.rho*self.pheromone_trace[ap,index_future] + (1-self.rho)*self.pheromone_trace[a,index_present] # move the pheromone in
                else:
                    self.pheromone_trace[a,index_present] = self.rho*self.pheromone_trace[a,index_present] + (1-self.rho)*self.pheromone_trace[ap,index_future] # move the pheromone in
                
                #always discount left state
                self.pheromone_trace[a,index_present] *= self.rho
               
                if done:
                    future_value = 0.0
                                
                delta = reward + self.gamma*future_value- present_value
                
                
                              
                self.tile_code_trace_weights[a,index_present] = 1
    
                self.tile_code_weights[a,index_present] = self.tile_code_weights[a,index_present] + self.lmbd*delta*self.alpha*self.tile_code_trace_weights[a,index_present] 
                self.tile_code_trace_weights= self.tile_code_trace_weights*self.gamma*self.lmbd
                
                a=ap
                s = sp
                
            if self.strategy=="TrueOnline":  
                                
                dict= {"state":s,"action": a,"time":t}
                self.current_trajectory[str(s*1.0)+"_"+str(a*1.0)] = dict
                
                sp, reward, done, _ = env.step(a)
                self.cum_reward = self.cum_reward + reward
                
                if self.model_based:      
                    ap= self.act2(sp)
                    
                else:
                    ap = self.act(sp)
                
                index_past = self.tile_coder[aold].__call__(sold)    
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
                
                #Pheromone part
                #move the pheromone away towards good trajectories, and discount it
                #the idea of this part really is to make the pheromone shape variable in the environment
                #pher_future_value = np.sum(self.select_state_action_weights(self.pheromone_trace, sp, ap))    
                #pher_present_value = np.sum(self.select_state_action_weights(self.pheromone_trace, s, a)) 
                #upwards more pheromone at the goal.
                
                #print np.sum((1-self.rho)*self.pheromone_trace[ap,index_future])#/self.num_tiling
                
                if self.heuristic_activate:


                    #Reasoning at the transition level, we have to take present, past and future
                    pher_past_value= np.sum(self.pheromone_trace[aold,index_past])
                    pher_present_value=np.sum(self.pheromone_trace[a,index_present])#
                    pher_future_value=np.sum(self.pheromone_trace[ap,index_future])#/(self.num_tiling+1))  
                    delta_present = ((pher_future_value+pher_present_value)/2 - (pher_present_value+pher_past_value)/2)/(self.num_tiling) #because we are in a link
                    
                    if delta_present > 0: #add a positive
                        self.pheromone_trace[aold,index_past] = self.rho*self.pheromone_trace[aold,index_past] + (1-self.rho)*delta_present
                        self.pheromone_trace[a,index_present] = self.rho*self.pheromone_trace[a,index_present] + (1-self.rho)*(delta_present)
                        #self.pheromone_trace[ap,index_future] = self.rho*self.pheromone_trace[ap,index_future] #- (1-self.rho)*(delta_present)
                    else: #subtract a negative
                        self.pheromone_trace[a,index_present] = self.rho*self.pheromone_trace[a,index_present] - (1-self.rho)*(delta_present)
                        self.pheromone_trace[ap,index_future] = self.rho*self.pheromone_trace[ap,index_future] - (1-self.rho)*(delta_present)
                    
                    self.pheromone_trace[a,index_present] = self.rho*self.pheromone_trace[a,index_present]
                    #self.pheromone_trace[ap,index_future] = self.rho*self.pheromone_trace[ap,index_future] - (1-self.rho)*delta_present
                
                
                    #else:
                        
                        #delta = pher_future_value -pher_present_value
                        #self.pheromone_trace[a,index_present] +=((1-self.rho)/self.num_tiling)*delta #self.pheromone_trace[a,index_future] # move the pheromone in
                        #self.pheromone_trace[ap,index_future] -=((1-self.rho)/self.num_tiling)*delta
                        #self.pheromone_trace[ap,index_future] *= self.rho
                        #self.pheromone_trace[a,index_present] *= self.rho
                aold = a
                sold = s
                a= ap
                s = sp
                
                
            
            if done: # something wrong in MC

                if self.cum_reward<0:
                    t = np.abs(self.cum_reward)
                else:
                    t=2 - 1/self.cum_reward #small impossible number

                self.last_steps.append(t)
                #if it is succesfful.
                result = self.success_invariant(t=t,env=env)
                
                
                if self.heuristic_activate:
                    self.handle_trajectory(result = result, t=t, trajectory= self.current_trajectory)
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
            return False
    
    
                
        
    def handle_trajectory(self, **args):
        t= args.get("t")
        result= args.get("result")
        Trajectory = args.get("trajectory")
        self.current_trajectory_to_ref= {"trajectory" : Trajectory, "quality" : 1.0/t, "success" : result} #

        if result:
            if 1.0/t > self.best_trajectory["quality"]:
                #print 1.0/t
                 
                self.best_trajectory= {"trajectory" : Trajectory, "quality" : 1.0/t, "success" : True} #
                #self.reinforce_heuristic_model()#Don't forget to reinforce in other models a heuristic of an empty vector is never going to work
                # it could be different here
                #one could think about storying multiple trajectories, rank them and enforce them... or else.
                
    
    def reinforce_heuristic_model(self,el=1):
        quality = self.best_trajectory["quality"]
        BTrajectory = self.best_trajectory["trajectory"]
        
        from multiprocessing.dummy import Pool
        
        self.pheromone_trace = np.dot(self.pheromone_trace,self.rho)
        
        
        quality2 = self.current_trajectory_to_ref["quality"]
        Trajectory = self.current_trajectory_to_ref["trajectory"]
        success = self.current_trajectory_to_ref["success"]
        
        
        #success is a Sanity check
        if success:
            if quality2 < quality:
                
                for val in Trajectory.values():
               
                    dict = val # this is a dict
                    s= dict["state"] # get the state
                    a= dict["action"]# get the action
                    time = dict["time"]
                    #pheromone update rule
               
                    index_pheromone_tiles = self.tile_coder[a].__call__(s)
               
                    self.pheromone_trace[a,index_pheromone_tiles] = self.pheromone_trace[a,index_pheromone_tiles] + quality2
                  
                   
        else: #if it is diverging, rather than recovering, try a hard reset. We still have the trajectories to learn from
                   #hard reset
            self.heuristic_activate = False
            #self.pheromone_trace = self.pheromone_trace*0
                   #self.pheromone_trace[a,index_pheromone_tiles] = 0 #self.pheromone_trace[a,index_pheromone_tiles] - cost2 # or to zero?
                   
            #self.tile_code_weights = self.tile_code_weights*0.0
            #self.tile_code_trace_weights = self.tile_code_trace_weights*0
                   #print "RESET!"
        
        #now set the pheromone for the best trajectory, if there was a failure, the good states will receive the pheromone
       
        
        for val in BTrajectory.values():
           
           dict = val # this is a dict
           s= dict["state"] # get the state
           a= dict["action"]# get the action
           time = dict["time"]
           #pheromone update rule
           
           index_pheromone_tiles = self.tile_coder[a].__call__(s) 
           self.pheromone_trace[a,index_pheromone_tiles] = self.pheromone_trace[a,index_pheromone_tiles] + quality
           #print self.pheromone_trace[a,index_pheromone_tiles]
    
    
        
        
    def dump_pher_trace_intra_episode(self,ep):           
        with open("pher_trace_intra_"+str(iteration)+".pkl","wb") as fh:
            pickle.dump(self.pheromone_trace,fh)
        with open("tile_coder_dyna.pkl","wb") as fh2:
            pickle.dump(self.tile_coder,fh2)
    
           
        #print "Reinforcing"   
       
                        
    def dump_pher_trace(self,FileName,FileNameTC):
        with open(FileName,"wb") as fh:
            pickle.dump(self.pheromone_trace,fh)
        with open(FileNameTC,"wb") as fh2:
            pickle.dump(self.tile_coder,fh2)
    
    