import gym
from gym import envs
from agents import TabularSarsaAgent
from agents import ApproximatedSarsaLambdaAgent
from agents import HAApproximatedSarsaLambdaAgent
from agents import PAApproximatedSarsaLambdaAgent
from agents import APHARSSarsaLambdaAgent 

from agents import StaticHeuristicApproximatedSarsaLambdaAgent 
from static_heuristics.LoadableHeuristic import LoadableHeuristic #import LoadableHeuristic
import numpy as np
import matplotlib.pyplot as plt
import pickle
from model import mc_model
import time

print(envs.registry.all())

env = gym.make('MountainCar-v0') #Defined by Stefano Bromuri

env._max_episode_steps = 1000

env.reset()

obs_mins = env.observation_space.low
obs_maxs = env.observation_space.high #[env.observation_space[0].max_value, env.observation_space[1].max_value]
discretizations = [10,10]

#Tabular Sarsa + Replacing Elegibility traces
#ag = TabularSarsaAgent.TabularSarsaAgent(discretize(),env.action_space)
#Tile coding Sarsa + Replacing Elegibility Traces
config = {  "Strategy" : "TrueOnline",
            "decrease_exploration" : True,
            "learning_rate" : 0.5,
            "eps": 0.025,            # Epsilon in epsilon greedy policies
            "lambda":0.9,
            "discount": 1,
            "n_iter": env._max_episode_steps} 

config_heur = {  "Strategy" : "TrueOnline",
                                  "Pheromone_strategy": "hard",
                                  "Plan_strategy": "soft",
                                  "decrease_exploration" : True, #Mountain Car has a decaying eploration
                                  "learning_rate" : 0.5,
                                  "psi": 0.0,
                                  "heuristic_dynamic":True,
                                  "model": mc_model.mc_model(),
                                  "model_based":True,
                                  "rho": 0.99,
                                  "eps": 0.025,
                                  "nu":0,            # Epsilon in epsilon greedy policies
                                  "lambda":0.9,
                                  "discount": 1,
                                  "n_stop":3, #Potential steps
                                  "n_iter": env._max_episode_steps} 




# rho, psi, nu, hard
# [0.0, 0.3, 0.0001, 1, [0.9], 0.5, 10]
# rho, psi, nu, soft
#[0.0, 0.9, 0.0001, 1, [0.9], 0.5, 5]

total_result = []
time_result= []

dict_res = {"series" : total_result,"times":time_result }

for j in range(100):
    print j
    
    #ag = ApproximatedSarsaLambdaAgent.ApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[10], my_config=config_heur)
    
    #Change the agent here.
    #ag = HAApproximatedSarsaLambdaAgent.HAApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[10], my_config=config_heur)
    #ag = PAApproximatedSarsaLambdaAgent.PAApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[10], my_config=config_heur)
    ag = APHARSSarsaLambdaAgent.APHARSSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[10], my_config=config_heur)
    
    #heur = MountainCarHeuristic(model= mc_model.mc_model(),actions_number=3)
    
    #heur = LoadableHeuristic("MountainCarPher.pkl","MountainCarTrace.pkl",actions_number=3)
    #config_heur["static_heuristic"] =heur
    #ag = StaticHeuristicApproximatedSarsaLambdaAgent.StaticHeuristicApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[10], my_config=config_heur)
    pher_trace = pickle.load( open( "MountainCarPher.pkl", "rb" ) )
    #ag.set_pheromone_trace(pher_trace)
    tc = pher_trace = pickle.load( open( "MountainCarTrace.pkl", "rb" ) )
    #ag.set_tilecoder(tc)
    
    
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
        print res[-1]
        
       
        #if i==199:
            #ag.dump_pher_trace("MountainCarPher.pkl","MountainCarTrace.pkl")
        
    total_result.append(res)
    time_result.append(times)
    dict_res["series"]=total_result
    dict_res["times"]=time_result
    with open("Mountain_car_simulation_true_online_AP-HARS-2.pkl", 'wb') as f:
        pickle.dump(dict_res, f)#write everything there
    
    
        #if i%10 == 0:
        #    plt.plot(res)
        #    plt.draw()
        #    plt.pause(0.0001)

 #   env.step(env.action_space.sample()) # take a random action