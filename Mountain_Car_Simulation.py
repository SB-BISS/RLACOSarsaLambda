import gym
from gym import envs
from agents import TabularSarsaAgent
from agents import ApproximatedSarsaLambdaAgent
from agents import HAApproximatedSarsaLambdaAgent
import numpy as np
import matplotlib.pyplot as plt
import pickle
from model import mc_model

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
                                  "decrease_exploration" : True, #Mountain Car has a decaying eploration
                                  "learning_rate" : 0.5,
                                  "psi": 0.3,
                                  "model": mc_model.mc_model(),
                                  "model_based":True,
                                  "rho": 0.99,
                                  "eps": 0.025,
                                  "nu":10,            # Epsilon in epsilon greedy policies
                                  "lambda":0.9,
                                  "discount": 1,
                                  "n_iter": env._max_episode_steps} 




# rho, psi, nu, hard
# [0.0, 0.3, 0.0001, 1, [0.9], 0.5, 10]
# rho, psi, nu, soft
#[0.0, 0.9, 0.0001, 1, [0.9], 0.5, 5]

total_result = []

for j in range(100):
    print j
    #Change the agent here.
    ag = HAApproximatedSarsaLambdaAgent.HAApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[10], my_config=config_heur)
    res = []
    for i in range(200):
        rend = False
        ag.learn(env,rend)
        res= ag.return_last_steps()
        print res[-1]
    total_result.append(res)
    with open("Mountain_car_simulation_true_online_heuristics_soft_tt.pkl", 'wb') as f:
        pickle.dump(total_result, f)
    
    
        #if i%10 == 0:
        #    plt.plot(res)
        #    plt.draw()
        #    plt.pause(0.0001)

 #   env.step(env.action_space.sample()) # take a random action