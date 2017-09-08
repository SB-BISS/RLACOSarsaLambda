import gym
from gym import envs
from agents import TabularSarsaAgent
from agents import ApproximatedSarsaLambdaAgent
from agents import HAApproximatedSarsaLambdaAgent
from agents import StaticHeuristicApproximatedSarsaLambdaAgent
import numpy as np
from model.dyna_model import dyna_model
from static_heuristics.EuclideanHeuristic import EuclideanHeuristic

import matplotlib.pyplot as plt
import pickle
from gym_maze.envs.maze_env import *

print(envs.registry.all())

env =  gym.make("dyna-v0")
env.seed(7)
 #Defined by Stefano Bromuri
episodes = 200
env._max_episode_steps = 5000

env.reset()

obs_mins = env.observation_space.low
obs_maxs = env.observation_space.high #[env.observation_space[0].max_value, env.observation_space[1].max_value]
discretizations = [20,20]

def test1():
    config_heur = {  "Strategy" : "TrueOnline",
                                      "Pheromone_strategy": "hard",
                                      "decrease_exploration" : False, #Mountain Car has a decaying eploration
                                      "learning_rate" : 0.1,
                                      "eps": 0.1,
                                      #"nu":1,            # Epsilon in epsilon greedy policies
                                      "lambda":0.9,
                                      "discount": 0.99,
                                      "n_iter": env._max_episode_steps} 
    
    # rho, psi, nu, hard
    # [0.3, 1e-05, 10]
    # rho, psi, nu, soft
    #[0.9, 1e-05, 5]
    
    total_result = []
    
    
    for j in range(100):
        print j
        #Change the agent here.
        ag = ApproximatedSarsaLambdaAgent.ApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[8], my_config=config_heur)
        res = []
        for i in range(episodes):
            rend = False
            ag.learn(env,rend)
            res= ag.return_last_steps()
            print res[-1]
        total_result.append(res)
        with open("Maze_simulation_true_online_no_heuristics.pkl", 'wb') as f:
            pickle.dump(total_result, f)


def test_heuristic():
    
    heur = EuclideanHeuristic(model= dyna_model(),goal=[25,25],actions_number=4)
    
    config_heur = {  "Strategy" : "TrueOnline",
                                      "decrease_exploration" : False, #Mountain Car has a decaying eploration
                                      "learning_rate" : 0.1,
                                      "eps": 0.1,
                                      #"nu":1,            # Epsilon in epsilon greedy policies
                                      "lambda":0.9,
                                      "discount": 0.99,
                                      "psi":0.1,
                                      "static_heuristic":heur,
                                      "nu":1,
                                      "n_iter": env._max_episode_steps} 
    
    # rho, psi, nu, hard
    # [0.3, 1e-05, 10]
    # rho, psi, nu, soft
    #[0.9, 1e-05, 5]
    
    total_result = []
    
    
    for j in range(100):
        print j
        #Change the agent here.
        ag = StaticHeuristicApproximatedSarsaLambdaAgent.StaticHeuristicApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[8], my_config=config_heur)
        res = []
        for i in range(episodes):
            rend = False
            ag.learn(env,rend)
            res= ag.return_last_steps()
            print res[-1]
        total_result.append(res)
        with open("Maze_simulation_true_online_static_heuristics.pkl", 'wb') as f:
            pickle.dump(total_result, f)










def test2():
    config_heur = {  "Strategy" : "TrueOnline",
                                      "Pheromone_strategy": "hard",
                                      "decrease_exploration" : False, #Mountain Car has a decaying eploration
                                      "learning_rate" : 0.1,
                                      "model_based":True,
                                      "model":dyna_model(),
                                      "psi": 0.3,
                                      "rho": 0.99,
                                      "eps": 0.1,
                                      "nu":1,            # Epsilon in epsilon greedy policies
                                      "lambda":0.9,
                                      "discount": 0.99,
                                      "n_iter": env._max_episode_steps} 
    
    # rho, psi, nu, hard
    # [0.3, 1e-05, 10]
    # rho, psi, nu, soft
    #[0.9, 1e-05, 5]
    
    total_result = []
    
    for j in range(100):
        print j
        #Change the agent here.
        ag = HAApproximatedSarsaLambdaAgent.HAApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[8], my_config=config_heur)
        res = []
        for i in range(episodes):
            rend = False
            ag.learn(env,rend)
            res= ag.return_last_steps()
            print res[-1]
        total_result.append(res)
        with open("Maze_simulation_true_online_heuristics_hard_model_based.pkl", 'wb') as f:
            pickle.dump(total_result, f)
        
def test3():
    config_heur = {  "Strategy" : "TrueOnline",
                                      "Pheromone_strategy": "hard",
                                      "decrease_exploration" : False, #Mountain Car has a decaying eploration
                                      "learning_rate" : 0.1,
                                      "model_based":False,
                                      "psi": 0.3,
                                      "rho": 0.01,
                                      "eps": 0.1,
                                      "nu":1,            # Epsilon in epsilon greedy policies
                                      "lambda":0.9,
                                      "discount": 0.99,
                                      "n_iter": env._max_episode_steps} 
    
    # rho, psi, nu, hard
    # [0.3, 1e-05, 10]
    # rho, psi, nu, soft
    #[0.9, 1e-05, 5]
    
    total_result = []
    
    for j in range(100):
        print j
        #Change the agent here.
        ag = HAApproximatedSarsaLambdaAgent.HAApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[8], my_config=config_heur)
        res = []
        for i in range(episodes):
            rend = False
            ag.learn(env,rend)
            res= ag.return_last_steps()
            print res[-1]
        total_result.append(res)
        with open("Maze_simulation_true_online_heuristics_hard2.pkl", 'wb') as f:
            pickle.dump(total_result, f)
               





#test1()
#test2()
#test3()
test_heuristic()