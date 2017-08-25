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
    Agent implementing approximated Sarsa-learning with traces.
    observation_space_mins = an ordered array with minimum values for each of the features of the exploration space
    observation_space_maxs = an ordered array with maximum values for each of the features of the exploation space
    num tiles = how many times do we split a dimension
    num tilings = how many overlapping tilings per dimension.
    
    Replacing traces model, typical replacing traces with tile coding
    TrueOnline Model, True online model!
    
    '''
    
    def __init__(self, observation_space_mins, observation_space_maxs, actions, num_tiles_l, num_tilings_l,  **userconfig):
        #if not isinstance(observation_space, discrete.Discrete):
        #    raise UnsupportedSpace('Observation space {} incompatible with {}. (Only supports Discrete observation spaces.)'.format(observation_space, self))
        #if not isinstance(action_space, discrete.Discrete):
        #    raise UnsupportedSpace('Action space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(action_space, self))
        super(self.__class__, self).__init__(observation_space_mins, observation_space_maxs, actions, num_tiles_l, num_tilings_l,  **userconfig)
        self.config['psi'] = 0.001
        self.config['rho'] = 0.9
        self.config['Pheromone_strategy'] = "Hard"
        self.config['nu'] =10
        conf = userconfig.get("my_config")
        self.config.update(conf)
        self.rho = self.config["rho"]
        self.nu = self.config["nu"]
        self.psi = self.config["psi"]
        self.strategy = self.config["Strategy"]
        print self.strategy
        # a trajectory contains state,action values history., plus the cost to pass back
        self.best_trajectory = {"trajectory" : [], "cost" : sys.maxint, "success" : False} # only succesfull trajectories should be enforced
        self.pheromone_trace = self.tile_code_trace_weights # this trace will include a pheromone value
   
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
        
        
        
        chance = np.random.random() 
        if chance> eps:
            action = np.argmax(possible_actions_values)
            action2 = np.argmax(possible_actions_values_pheromones)
            
       
            val = possible_actions_values[action]
            val2 = possible_actions_values_pheromones[action2]
            
            difference_term = self.psi*(val-val2 +1) #this plus 1 is nu
            possible_actions_values[action2] = possible_actions_values[action2] + difference_term
            action = np.argmax(possible_actions_values)
            
            q2 = action
            
            
        else:
            action=self.action_space.sample()
            #action = np.random.randint(0,3)
        return action
    
    

    def learn(self, env, rend = False):
        config = self.config
        s = env.reset()
        
        #tiles = self.tile_coder
        #self.trace= self.create_state_action_table() # reset
        #trace = self.trace
        #self.tile_coder_traces.reset() # reset replacing traces
  
        self.tile_code_trace_weights = self.tile_code_trace_weights*0
        #traces = self.tile_coder_traces
        
        Trajectory = []
        
        for t in range(config["n_iter"]):
            
            
            
            if self.decrease_exploration:
                self.config["eps"] = self.config["eps"]*0.99
                
            
            #print s
            
            if self.strategy=="Replacing":
               
                a = self.act(s)
                dict = {"state":s,"action":a}
                Trajectory.append(dict)
                
                sp, reward, done, _ = env.step(a)
                future = 0.0
            
                #if not done:
                    #future = np.max(q[obs2.item()])
                ap= self.act(sp)
                
                index_present = self.tile_coder[a].__call__(s)
                index_future = self.tile_coder[ap].__call__(sp)
                
                
                
                future_value = np.sum(self.select_state_action_weights(self.tile_code_weights, sp, ap))    
                present_value = np.sum(self.select_state_action_weights(self.tile_code_weights, s, a)) 
                
               
                
                delta = reward + self.gamma*future_value- present_value
                #print delta
                self.tile_code_trace_weights[a,index_present] = 1
                
                self.tile_code_weights[a,index_present] = self.tile_code_weights[a,index_present] + self.lmbd*delta*self.alpha*self.tile_code_trace_weights[a,index_present] 
                self.tile_code_trace_weights= self.tile_code_trace_weights*self.gamma*self.lmbd
                
                s = sp
                
            if self.strategy=="TrueOnline":  
                
                
                a = self.act(s)
                
                Trajectory.append({"state":s,"action":a})
                
                sp, reward, done, _ = env.step(a)
                future = 0.0
            
                ap= self.act(sp)
                
                index_present = self.tile_coder[a].__call__(s)
                index_future = self.tile_coder[ap].__call__(sp)
                 
                #same indices, different weights
                future_value = np.sum(self.select_state_action_weights(self.tile_code_weights, sp, ap))    
                present_value = np.sum(self.select_state_action_weights(self.tile_code_weights, s, a)) 
                present_value_old = np.sum(self.select_state_action_weights(self.tile_code_weights_old, s, a)) # calculated here
               
                delta = reward + self.gamma*future_value- present_value
                
                self.tile_code_weights_old = self.tile_code_weights # before it changes, but after I took the value of present_value_old
                
                
                self.tile_code_trace_weights[a,index_present] = self.tile_code_trace_weights[a,index_present] + self.alpha*(1- self.lmbd*self.gamma*self.tile_code_trace_weights[a,index_present]) # add value to the cell 
                
                self.tile_code_weights[a,index_present] = self.tile_code_weights[a,index_present] + self.lmbd*delta*self.alpha*self.tile_code_trace_weights[a,index_present] + self.alpha*(present_value-present_value_old)
                self.tile_code_trace_weights= self.tile_code_trace_weights*self.gamma*self.lmbd # discount all the old values after application
                
                s = sp

            
            if done: # something wrong in MC
                self.last_steps.append(t)
                
                #if it is succesfful.
                result = self.success_invariant(t=t,env=env)
                if result:
                    self.handle_trajectory(result = result, t=t, trajectory=Trajectory)
                    
                #print s
                break
            
            if rend:
                env.render()
                
    
    #to be overridden according to the scenario, if no overriding needed just don't
    #change it in the subclass
    #in this case I consider a trajectory succesfull if
    def success_invariant(self,**args):
        t= args.get("t")
        env = args.get("env")
                 
        if t<env._max_episode_steps:
            return True
        else:
            return False
        
    def handle_trajectory(self, **args):
        t= args.get("t")
        result= args.get("result")
        Trajectory = args.get("trajectory")
        if result:
            if 1/t < self.best_trajectory["cost"]:
                self.best_trajectory= {"trajectory" : Trajectory, "cost" : 1/t, "success" : True} # it could be different here
                #one could think about storying multiple trajectories, rank them and enforce them... or else.
                
    
    def reinforce_heuristic_model(self):
        cost = self.best_trajectory["cost"]
        for i in self.best_trajectory["trajectory"]:
           s= i["state"] # get the state
           a= i["action"]# get the action
           #pheromone update rule
           
           index_pheromone_tiles = self.tile_coder[a].__call__(s)
           
           self.pheromone_trace[a,index_pheromone_tiles] = self.pheromone_trace[a,index_pheromone_tiles] + cost
           self.pheromone_trace = self.rho*self.pheromone_trace
                        
'''
                
function [Trace,StructTiling] = UpdateQLearningTilesMountainCarNoHashingTrueOnline(StructTiling, r, alpha, gamma,lambda,position, newposition, Trace, Traj )
% UpdateQ update de Qtable and return it using Whatkins Q-Learing
% s1: previous state before taking action (a)
% s2: current state after action (a)
% r: reward received from the environment after taking action (a) in state
%                                             s1 and reaching the state s2
% a:  the last executed action
% tab: the current Qtable
% alpha: learning rate
% gamma: discount factor
% Q: the resulting Qtable
% UpdateQ update de Qtable and return it using Whatkins Q-Learing
% s1: previous state before taking action (a)
% s2: current state after action (a)
% r: reward received from the environment after taking action (a) in state
%                                             s1 and reaching the state s2
% a:  the last executed action
% tab: the current Qtable
% alpha: learning rate
% gamma: discount factor
% Q: the resulting Qtable

   
    
   Tiles =   StructTiling.tc;  
   TilesWeight = StructTiling.tw;
   TilesWeightOld = StructTiling.twold;
   
   
   StructTiling.twold = StructTiling.tw;

   TilingInfo = StructTiling.ti;
   
  %old
  [vali idxtilei ] = tiling_get(Tiles,TilesWeightOld,TilingInfo,position);
  
  %current
  [valii idxtileii ] = tiling_get(Tiles,TilesWeight,TilingInfo,position);
  
  %next
  [val2 idxtile2 ] = tiling_get(Tiles,TilesWeight,TilingInfo,newposition);
   
  %position
  %newposition

    NeT =  val2; %sum(WeightsNew);
    Neti = vali;
    Netii = valii;
    
   % if ismember(A(1),BestPerformingTiles)
    
   %     r = -2;
   % end    
    
   %if ~isempty(SF)  
    %ThetaError = (r+ gamma*NeT - Old*(Theta-SF)');
    %ThetaError
   %else
    ThetaError = (r  + (gamma*NeT - Neti));  
    
   %end    
   
       
    Trace = gamma*lambda*Trace;
    Trace(idxtilei) = Trace(idxtilei)+  (alpha)*(1  - lambda*gamma*Trace(idxtilei));
    
    
    
    %Theta = Theta + ((alpha/features_n)*ThetaError).*Trace; %SF(A);
    
    %Theta(A) = Theta(A) + ((alpha/features_n)*ThetaError).*Trace(A); %SF(A);
    
    %[v, i, j, wdata]  = tiling_update(Tiles,TilesWeight,TilingInfo,position,(alpha/features_n)*ThetaError,0);
    % idxtile
    %size(TilesWeight(idxtile))
    %size(Trace(idxtile))
    %size(ThetaError)
    
    TilesWeight(idxtilei) = TilesWeight(idxtilei) + ThetaError*Trace(idxtilei) +alpha*(Neti -Netii);
 
    
    StructTiling.tw = TilesWeight;
  
    
    
    
end    
    %Theta
    
%TD_error =   ((r + gamma*max(Q(sp,:))) - Q(s,a));
%Q(s,a) =  Q(s,a) + alpha * TD_error;
'''                
    