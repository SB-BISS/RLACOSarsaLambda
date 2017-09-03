from gym.spaces import discrete
import gym
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import pdist, squareform

class TabularSarsaAgent(object):
    """
    Agent implementing tabular Sarsa-learning.
    The observation space must be discretized according to the environment being studied.
    
    """
    
    def __init__(self, observation_space, action_space, **userconfig):
        #if not isinstance(observation_space, discrete.Discrete):
        #    raise UnsupportedSpace('Observation space {} incompatible with {}. (Only supports Discrete observation spaces.)'.format(observation_space, self))
        #if not isinstance(action_space, discrete.Discrete):
        #    raise UnsupportedSpace('Action space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(action_space, self))
        self.observation_space = observation_space
        self.action_space = action_space
        self.last_steps = []
        
        #delta  =  ( r + gamma * Q(sp,ap) ) - Q(s,a);
        #%trace(s,:) = 0.0;  %optional trace reset
        #trace(s,a) = 1.0;

        #Q = Q + alpha * delta * trace;

        #trace = gamma * lambda * trace;

        
        self.action_n = action_space.n
        self.config = {
            "decrease_exploration" : True,
            "learning_rate" : 0.3,
            "eps": 0.01,            # Epsilon in epsilon greedy policies
            "lambda":0.9,
            "discount": 1,
            "n_iter": 500}        # Number of iterations
        self.config.update(userconfig)
        self.alpha = self.config["learning_rate"]
        self.lmbd = self.config["lambda"]
        self.gamma = self.config["discount"]
        self.decrease_exploration = self.config["decrease_exploration"]
        
        self.q = self.create_state_action_table()  #np.zeros((self.config["n_states"],self.config["n_states"],self.action_n)) 
        self.trace= self.create_state_action_table()# just the same
        #defaultdict(lambda: self.config["init_std"] * np.random.randn(self.action_n) + self.config["init_mean"])

    def create_state_action_table(self):
        nstates     = np.shape(self.observation_space)[0]
        nactions    = self.action_space.n
        Q = [[0.0 for i in range(nactions)] for i in range(nstates)]
        return Q


    def act(self, observation, eps=None):
        if eps is None:
            eps = self.config["eps"]
        # epsilon greedy.
        
        state = self.discretize_state(observation) 
        
        chance = np.random.random() 
        if chance> eps:
            action = np.argmax(self.q[state])
        else:
            action=self.action_space.sample()
            #action = np.random.randint(0,3)
        return action
    
    
    def discretize_state(self,x):    
        """DiscretizeState check which entry in the state list is more close to x and return the index of that entry."""
       
        space = self.observation_space 
        space = np.vstack((space,x))
        Md= squareform(pdist(space))
      
        last = Md[-1] #last row
        #print len(last)
        #print last
        #print np.argmin(last)
        #print np.argmin(last[0:-1])
        #print np.argmin(last[0:-2])
        #print np.argmin(last[0:-2])
        return  np.argmin(last[0:-1])    
    
    
    def return_last_steps(self):
        return self.last_steps
    #the state here has multiple variables.

    def learn(self, env, rend = False):
        config = self.config
        s = env.reset()
        q = self.q
        self.trace= self.create_state_action_table() # reset
        trace = self.trace
        action = self.act(s)
        
        for t in range(config["n_iter"]):
            
            if self.decrease_exploration:
                self.config["eps"] = self.config["eps"]*0.99
            
            
            sp, reward, done, _ = env.step(action)
            future = 0.0
            
            current_state = self.discretize_state(s)
           
            #if not done:
                #future = np.max(q[obs2.item()])
            actionp= self.act(sp)
    
            future_state = self.discretize_state(sp)
            future = q[future_state][actionp]
            
            present= q[current_state][action]
            
            #delta
            delta= (reward + self.gamma * future - present)
            trace[current_state][action] = 1.0
            
            q = q + self.alpha*delta*np.array(trace) 
            trace= self.gamma*self.lmbd*np.array(trace)
            self.q = q
            self.trace = trace
            
            s = sp
            action = actionp
            if done: # something wrong in MC
                self.last_steps.append(t)
                print s
                break
            
            if rend:
                env.render()