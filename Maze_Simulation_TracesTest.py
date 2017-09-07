import gym
from gym import envs
from agents import TabularSarsaAgent
from agents import ApproximatedSarsaLambdaAgent
from agents import HAApproximatedSarsaLambdaAgent
import numpy as np
import matplotlib.pyplot as plt
import pickle
from gym_maze.envs.maze_env import *

print(envs.registry.all())

env =  gym.make("maze-sample-20x20-v0")
 #Defined by Stefano Bromuri
episodes = 200
env._max_episode_steps = 10000

env.reset()

obs_mins = env.observation_space.low
obs_maxs = env.observation_space.high #[env.observation_space[0].max_value, env.observation_space[1].max_value]
discretizations = [20,20]


def test1():
    config_heur = {  "Strategy" : "TrueOnline",
                                      "Pheromone_strategy": "hard",
                                      "decrease_exploration" : False, #Mountain Car has a decaying eploration
                                      "learning_rate" : 0.1,
                                      "psi": 0.00001,
                                      "rho": 0.1,
                                      "eps": 0.1,
                                      "nu":1,            # Epsilon in epsilon greedy policies
                                      "lambda":0.01,
                                      "discount": 0.9,
                                      "n_iter": env._max_episode_steps} 
    
    # rho, psi, nu, hard
    # [0.3, 1e-05, 10]
    # rho, psi, nu, soft
    #[0.9, 1e-05, 5]
    
    total_result = []
    
    for j in range(100):
        print j
        #Change the agent here.
        ag = ApproximatedSarsaLambdaAgent.ApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[10], my_config=config_heur)
        res = []
        for i in range(episodes):
            rend = False
            ag.learn(env,rend)
            res= ag.return_last_steps()
            print res[-1]
        total_result.append(res)
        with open("Maze_simulation_true_online_no_heuristics.pkl", 'wb') as f:
            pickle.dump(total_result, f)


def test2():
    config_heur = {  "Strategy" : "TrueOnline",
                                      "Pheromone_strategy": "hard",
                                      "decrease_exploration" : False, #Mountain Car has a decaying eploration
                                      "learning_rate" : 0.1,
                                      "psi": 0.3,
                                      "rho": 0.99,
                                      "eps": 0.1,
                                      "nu":10,            # Epsilon in epsilon greedy policies
                                      "lambda":0.01,
                                      "discount": 0.9,
                                      "n_iter": env._max_episode_steps} 
    
    # rho, psi, nu, hard
    # [0.3, 1e-05, 10]
    # rho, psi, nu, soft
    #[0.9, 1e-05, 5]
    
    total_result = []
    
    for j in range(100):
        print j
        #Change the agent here.
        ag = HAApproximatedSarsaLambdaAgent.HAApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[10], my_config=config_heur)
        res = []
        for i in range(episodes):
            rend = False
            ag.learn(env,rend)
            res= ag.return_last_steps()
            print res[-1]
        total_result.append(res)
        with open("Maze_simulation_true_online_heuristics_hard.pkl", 'wb') as f:
            pickle.dump(total_result, f)
        
def test3():
    config_heur = {  "Strategy" : "TrueOnline",
                                      "Pheromone_strategy": "soft",
                                      "decrease_exploration" : False, #Mountain Car has a decaying eploration
                                      "learning_rate" : 0.1,
                                      "psi": 0.3,
                                      "rho": 0.99,
                                      "eps": 0.1,
                                      "nu":10,            # Epsilon in epsilon greedy policies
                                      "lambda":0.01,
                                      "discount": 0.9,
                                      "n_iter": env._max_episode_steps} 
    
    # rho, psi, nu, hard
    # [0.3, 1e-05, 10]
    # rho, psi, nu, soft
    #[0.9, 1e-05, 5]
    
    total_result = []
    
    for j in range(100):
        print j
        #Change the agent here.
        ag = HAApproximatedSarsaLambdaAgent.HAApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[10], my_config=config_heur)
        res = []
        for i in range(episodes):
            rend = False
            ag.learn(env,rend)
            res= ag.return_last_steps()
            print res[-1]
        total_result.append(res)
        with open("Maze_simulation_true_online_heuristics_hard.pkl", 'wb') as f:
            pickle.dump(total_result, f)
               





#test1()
#test2()
test2()
