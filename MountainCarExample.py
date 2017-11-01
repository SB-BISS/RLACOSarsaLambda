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
layers = 10

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
                                  "psi": 0.1,
                                  "heuristic_dynamic":True,
                                  "model": mc_model.mc_model(),
                                  "model_based":False,
                                  "rho": 0.99,
                                  "eps": 0.025,
                                  "nu":10,            # Epsilon in epsilon greedy policies
                                  "lambda":0.9,
                                  "discount": 1,
                                  "n_stop":3, #Potential steps
                                  "n_iter": env._max_episode_steps} 



total_result = []
time_result= []

dict_res = {"series" : total_result,"times":time_result }

print env.action_space
print env.observation_space

#ag = ApproximatedSarsaLambdaAgent.ApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[10], my_config=config_heur)
ag = HAApproximatedSarsaLambdaAgent.HAApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[layers], my_config=config_heur)

    
res = []
times = []
for i in range(200):
    rend = False
    if i == 100:
        rend = True

    time_before = time.time()
    ag.learn(env,rend)
    res= ag.return_last_steps()
    time_after = time.time()
    time_req = time_after - time_before
    times.append(time_req)



    #if i % 10 == 0:
    plt.plot(res)
    plt.draw()
    plt.show(block=False)
    plt.pause(0.0001)

    print res[-1]

