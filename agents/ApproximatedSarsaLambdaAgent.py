from gym.spaces import discrete
import gym
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tilecoding.representation import TileCoding

class ApproximatedSarsaLambdaAgent(object):
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
       
        self.last_steps = []
        self.action_space = actions
        self.action_n = actions.n
        self.config = {
            "Strategy" : "Replacing",
            "decrease_exploration" : True,
            "learning_rate" : 0.5,
            "eps": 0.025,            # Epsilon in epsilon greedy policies
            "lambda":0.9,
            "discount": 1,
            "n_iter": 500}        # Number of iterations
        
        my_config = userconfig.get("my_config")
        
        self.config.update(my_config)
        
        
        
        self.strategy = self.config["Strategy"]
        self.alpha = self.config["learning_rate"]/num_tilings_l[0] # assumption: we have one costant for the number of tilings
        self.lmbd = self.config["lambda"]
        self.gamma = self.config["discount"]
        self.decrease_exploration = self.config["decrease_exploration"]
        
        self.tile_coder = {}
        self.tile_coder_traces = {}
        
        size_tc = 0
        size_tct = 0 
        
        for i in range(0,actions.n):
            
            state_range= [observation_space_mins,observation_space_maxs]
            self.tile_coder[i]=TileCoding(input_indices = [np.arange(len(observation_space_mins))], 
                        ntiles = num_tiles_l, 
                        ntilings = num_tilings_l, 
                        hashing = None,
                        state_range = state_range,
                        rnd_stream = np.random.RandomState())
            
            size_tc = self.tile_coder[i].size
            
            self.tile_coder_traces[i]=TileCoding(input_indices = [np.arange(len(observation_space_mins))], 
                        ntiles = num_tiles_l, ntilings = num_tilings_l, hashing = None,
                        state_range = state_range,
                        rnd_stream = np.random.RandomState())
            
            size_tct = self.tile_coder_traces[i].size
        
        self.tile_code_weights = np.zeros((actions.n,size_tc))
        self.tile_code_weights_old = np.zeros((actions.n,size_tc)) # for true online traces
        self.tile_code_trace_weights = np.zeros((actions.n,size_tct))  
        
   
    def select_state_action_weights(self,vector_weights,observation,action):
        return vector_weights[action,self.tile_coder[action].__call__(observation)] # select indices, 
    
    def select_trace_weights(self,vector_weights,observation,action):
        return vector_weights[action,self.tile_coder_traces[action].__call__(observation)] # select indices
    
    
        
    def act(self, observation, eps=None):
        if eps is None:
            eps = self.config["eps"]
        # epsilon greedy.
        
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
    
    
    def return_last_steps(self):
        return self.last_steps
    #the state here has multiple variables.

    def learn(self, env, rend = False):
        config = self.config
        s = env.reset()
        
        #tiles = self.tile_coder
        #self.trace= self.create_state_action_table() # reset
        #trace = self.trace
        #self.tile_coder_traces.reset() # reset replacing traces
        present_value_old = 0
        
        if self.strategy=="Replacing":
            self.tile_code_trace_weights = self.tile_code_trace_weights*0
        #traces = self.tile_coder_traces
        
        
        a = self.act(s)
        
        for t in range(config["n_iter"]):
            
            if self.decrease_exploration:
                self.config["eps"] = self.config["eps"]*0.99
            
            #print s
            
            if self.strategy=="Replacing":
            
                sp, reward, done, _ = env.step(a)
                future = 0.0
            
                #if not done:
                    #future = np.max(q[obs2.item()])
                ap= self.act(sp)
                
                index_present = self.tile_coder[a].__call__(s)
                index_future = self.tile_coder[ap].__call__(sp)
                
                future_value = np.sum(self.select_state_action_weights(self.tile_code_weights, sp, ap))    
                present_value = np.sum(self.select_state_action_weights(self.tile_code_weights, s, a)) 
                
                if done:
                    future_value = 0.0
                
                delta = reward + self.gamma*future_value- present_value
                #print delta
                self.tile_code_trace_weights[a,index_present] = 1
                
                self.tile_code_weights[a,index_present] = self.tile_code_weights[a,index_present] + self.lmbd*delta*self.alpha*self.tile_code_trace_weights[a,index_present] 
                self.tile_code_trace_weights= self.tile_code_trace_weights*self.gamma*self.lmbd
                
                #self.tile_coder = tiles
                #self.tile_coder_traces = traces
                
                s = sp
                a= ap
                
            if self.strategy=="TrueOnline":  
                
                sp, reward, done, _ = env.step(a)
                #future = 0.0
            
                #if not done:
                    #future = np.max(q[obs2.item()])
                ap= self.act(sp)
                
                #index_old_present = self.tile_coder_old[a].__call__(s) # necessary for true online traces
                index_present = self.tile_coder[a].__call__(s)
                index_future = self.tile_coder[ap].__call__(sp)
                
                      #same indices, different weights
                present_value = np.sum(self.select_state_action_weights(self.tile_code_weights, s, a)) 
                future_value = np.sum(self.select_state_action_weights(self.tile_code_weights, sp, ap))
                
                if done:
                    future_value = 0.0 #Very important in TrueOnline.   
              
                dutch = (1 -self.alpha*self.gamma*self.lmbd*np.sum(self.tile_code_trace_weights[a,index_present])) # just a value.
                #print dutch
                self.tile_code_trace_weights[a,index_present] += dutch
               
                delta = reward + self.gamma*future_value- present_value
                delta_q = (present_value - present_value_old)
                
                #self.tile_code_weights +=  self.alpha * (delta + present_value - present_value_old) * self.tile_code_trace_weights
                self.tile_code_weights +=  self.alpha * (delta) * self.tile_code_trace_weights  +  self.alpha*(delta_q)*self.tile_code_trace_weights 
                self.tile_code_weights[a,index_present] = self.tile_code_weights[a,index_present]- self.alpha *(delta_q)
                
                self.tile_code_trace_weights= self.tile_code_trace_weights*self.gamma*self.lmbd 
                
             
                #       
                present_value_old = future_value
                
                s = sp
                a= ap
            
            
            if done: # something wrong in MC
                self.last_steps.append(t)
                #print s
                break
            
            if rend:
                env.render()
                
                
                
                
                
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
    