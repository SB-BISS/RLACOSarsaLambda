import gym
from gym import envs
from agents import TabularSarsaAgent
from agents import ApproximatedSarsaLambdaAgent
from agents import StaticHeuristicApproximatedSarsaLambdaAgent
from agents import HAApproximatedSarsaLambdaAgent
from static_heuristics.M3DHeuristic import M3DHeuristic 
from static_heuristics.LoadableHeuristic import LoadableHeuristic #import LoadableHeuristic

import numpy as np
import matplotlib.pyplot as plt
import pickle
from gym_m3d.envs.m3d_environment import *
from model.m3d_model import m3d_model
import time

print(envs.registry.all())

env = gym.make('m3d-v0') #Defined by Stefano Bromuri

env._max_episode_steps = 1000

env.reset()

obs_mins = env.observation_space.low
obs_maxs = env.observation_space.high #[env.observation_space[0].max_value, env.observation_space[1].max_value]
discretizations = [4,4,4,4]


config_heur = {  "Strategy" : "TrueOnline",
                                  "Pheromone_strategy": "hard",
                                  "decrease_exploration" : True,
                                  "descrease_exploration_rate": 0.9, #Mountain Car has a decaying eploration
                                  "learning_rate" : 0.5,
                                  "psi": 0.01,
                                  "rho": 0.99,
                                  "eps": 0.5,
                                  "model": m3d_model(),
                                  "model_based": True,
                                  "nu":5,            # Epsilon in epsilon greedy policies
                                  "lambda":0.9,
                                  "discount": 1,
                                  "n_iter": env._max_episode_steps} 




# rho, psi, nu, hard
# [0.3, 1e-05, 10]
# rho, psi, nu, soft
#[0.9, 1e-05, 5]

total_result = []
time_result= []

dict_res = {"series" : total_result,"times":time_result }

for j in range(100):
    print j
    #Change the agent here.
    heur = M3DHeuristic(model= m3d_model(),actions_number=5)
    #heur = heur = LoadableHeuristic("M3DPher.pkl","M3DTrace.pkl",actions_number=5)
    config_heur["static_heuristic"] =heur
    
    #ag = HAApproximatedSarsaLambdaAgent.HAApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[10], my_config=config_heur)
    ag = StaticHeuristicApproximatedSarsaLambdaAgent.StaticHeuristicApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[10], my_config=config_heur)
    
    #ag = ApproximatedSarsaLambdaAgent.ApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[5], my_config=config_heur)
    
    
    res = []
    times = []
    for i in range(200):
        rend = False
        
        time_before = time.time()
        ag.learn(env,rend)
        res= ag.return_last_steps()
        time_after = time.time()
        time_req = time_after - time_before
        times.append(time_req)
        
        #if i==199:
        #    ag.dump_pher_trace("M3DPher.pkl","M3DTrace.pkl")
        
        print res[-1]
    total_result.append(res)
    time_result.append(times)
    dict_res["series"]=total_result
    dict_res["times"]=time_result
    with open("M3D_simulation_true_online_static_heuristic2f.pkl", 'wb') as f:
        pickle.dump(dict_res, f)#write everything there
    
    
        #if i%10 == 0:
        #    plt.plot(res)
        #    plt.draw()
        #    plt.pause(0.0001)

 #   env.step(env.action_space.sample()) # take a random action