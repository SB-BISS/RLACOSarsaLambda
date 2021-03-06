'''
This is a GRID search to find the best parameters of an algorithm.
In future developments it will be parallelized.

In the Mountain Car Problem, for Sarsa with Tile Coding, we 
have the parameters from Sutton and Barto Book

alpha = 0.5 (then divided by num tilings, so it becomes 0.5/0.8, check the implementation of the agents)
decaying factor = 0.96
lambda = 0.96
Discretization = 8,8

'''




import gym
from gym import envs
from agents import TabularSarsaAgent
from agents import ApproximatedSarsaLambdaAgent
from agents import HAApproximatedSarsaLambdaAgent
from agents import StaticHeuristicApproximatedSarsaLambdaAgent
from static_heuristics.EuclideanHeuristic import EuclideanHeuristic 
from model.dyna_model import dyna_model

import numpy as np
import matplotlib.pyplot as plt
from gym_maze.envs.maze_env import *
import time
from model import m3d_model


import pickle

print(envs.registry.all())

env =  gym.make("dyna-v0")
env.seed(7)

env._max_episode_steps = 5000
repetitions = 25
episodes = 50

env.reset()

obs_mins = env.observation_space.low
obs_maxs = env.observation_space.high #[env.observation_space[0].max_value, env.observation_space[1].max_value]
print obs_mins
print obs_maxs
discretizations = [20,20] #position 
num_tilings = 10

total_result = []
rend = False # render or not.
#values for Rho
rho_pos = [0.1,0.3,0.6,0.9,0.99]  # [0.1,0.5,0.99] #3
#values for psi, for the heuristic
psi_pos = [0.001, 0.01,0.1,0.3,0.5]  # [0.00001, 0.0001,0.001,0.01,0.1,1] # 5
#values of nu, for the heuristic
nu_pos = [1,5,10] 

#values for discount factor
discount_pos = [0.99] # not discounted
lmbd = [0.9]# Lambda get the same value...
alpha_pos = [0.1] #it becomes 0.5/8, given the num tilings above
eps_pos = [0.1] #decaying exploration

# one iteration of the grid search

algorithms = ["NOH","SH","H"]
Strategies = ["Replacing","TrueOnline"]

algo = algorithms[1]
strat = Strategies[1]
hard_soft = "hard"
model_based = False

if algo == "SH":
    rho_pos = [0.0]


z= 0 #counter

for eps in eps_pos:
    for rho in rho_pos:
        for psi in psi_pos:
            for dis in discount_pos: 
                for nu in nu_pos:
                    for alpha in alpha_pos:
                        
                        config = {  "Strategy" : strat,
                                  "Pheromone_strategy": hard_soft,
                                  "decrease_exploration" : False, #Mountain Car has a decaying eploration
                                  "learning_rate" : alpha,
                                  "psi": psi,
                                  "rho": rho,
                                  "static_heuristic": EuclideanHeuristic(model= dyna_model(),goal=[25,25],actions_number=4),
                                  "model_based":model_based,
                                  "eps": eps,
                                  "nu":nu,            # Epsilon in epsilon greedy policies
                                  "lambda":lmbd[0],
                                  "discount": dis,
                                  "n_iter": env._max_episode_steps} 

                        
                        
                        times = np.zeros(episodes)
                        results = np.zeros(episodes)
                        print z
                        for j in range(repetitions): # this is to decide for the parameter
                            #env.seed(7)

                            if algo=="NOH":
                                 ag = ApproximatedSarsaLambdaAgent.ApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[num_tilings], my_config=config)
                            
                            elif algo=="SH":
                                 ag = StaticHeuristicApproximatedSarsaLambdaAgent.StaticHeuristicApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[num_tilings], my_config=config)
                            
                            else:
                                ag = HAApproximatedSarsaLambdaAgent.HAApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[num_tilings], my_config=config)
                            
                            for i in range(episodes):
                                
                                tb = time.time()
                                ag.learn(env,rend)
                                te = time.time()
                                tdiff= te-tb
                                res= ag.return_last_steps()
                                results[i] = results[i]+res[i]
                                print res[i]
                                times[i] = times[i] + tdiff
                                  
                                print i
                                #print (res[-1], [eps,rho,psi,dis,dis,alpha])
                        #in the maze grid search you are looking for the one with the smallest cumulative_sum        
                        total_result.append({"parameters": [eps,rho,psi,dis,lmbd,alpha,nu] , "times":times/repetitions, "20thep": results[-1]/repetitions, "results":results/repetitions, "cumulative_sum": np.sum(results/repetitions)})
                         #   env.step(env.action_space.sample()) # take a random action
                        z = z+1
                        with open("Dyna_"+algo+"_" + strat + "_" + hard_soft + "model"+str(model_based)+ ".pkl", 'wb') as f:
                            pickle.dump(total_result, f)
                            

 
#Saving the result of the GRID Search 

 
 
 