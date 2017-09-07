import gym
from gym import envs
from agents import TabularSarsaAgent
from agents import ApproximatedSarsaLambdaAgent
from agents import HAApproximatedSarsaLambdaAgent
import numpy as np
import matplotlib.pyplot as plt
import pickle
from gym_m3d.envs.m3d_environment import *
from model.m3d_model import m3d_model

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
                                  "psi": 0.3,
                                  "rho": 0.99,
                                  "eps": 0.5,
                                  "model": m3d_model(),
                                  "model_based": False,
                                  "nu":10,            # Epsilon in epsilon greedy policies
                                  "lambda":0.9,
                                  "discount": 1,
                                  "n_iter": env._max_episode_steps} 




# rho, psi, nu, hard
# [0.3, 1e-05, 10]
# rho, psi, nu, soft
#[0.9, 1e-05, 5]

total_result = []

for j in range(100):
    print j
    #Change the agent here.
    ag = HAApproximatedSarsaLambdaAgent.HAApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[5], my_config=config_heur)
    res = []
    for i in range(200):
        rend = False
        ag.learn(env,rend)
        res= ag.return_last_steps()
        print res[-1]
    total_result.append(res)
    with open("M3D_simulation_true_online_heuristics_hard.pkl", 'wb') as f:
        pickle.dump(total_result, f)
    
    
        #if i%10 == 0:
        #    plt.plot(res)
        #    plt.draw()
        #    plt.pause(0.0001)

 #   env.step(env.action_space.sample()) # take a random action