import gym
from gym import envs
from agents import TabularSarsaAgent
from agents import ApproximatedSarsaLambdaAgent
from agents import HAApproximatedSarsaLambdaAgent
from agents import APHARSSarsaLambdaAgent
from agents import PAApproximatedSarsaLambdaAgent
from agents import StaticHeuristicApproximatedSarsaLambdaAgent
import numpy as np
import time
from model.dyna_model import dyna_model
from static_heuristics.EuclideanHeuristic import EuclideanHeuristic
from static_heuristics.LoadableHeuristic import LoadableHeuristic



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
    time_result= []
    dict_res = {"series" : total_result,"times":time_result }
    
    for j in range(100):
        print j
        #Change the agent here.
        ag = ApproximatedSarsaLambdaAgent.ApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[10], my_config=config_heur)
        res = []
        times = []

        for i in range(episodes):
            rend = False
            time_before = time.time()
            ag.learn(env,rend)
            time_after = time.time()
            time_req = time_after - time_before
            times.append(time_req)
            res= ag.return_last_steps()
            print res[-1]
        total_result.append(res)
        time_result.append(times)
        dict_res["series"]=total_result
        dict_res["times"]=time_result
        with open("TXT1.pkl", 'wb') as f:
            pickle.dump(dict_res, f)


def test_heuristic():
    
    #heur = EuclideanHeuristic(model= dyna_model(),goal=[25,25],actions_number=4)
    heur = LoadableHeuristic("DynaPher.pkl","DynaTC.pkl",actions_number=4)
    
    
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
    time_result= []
    dict_res = {"series" : total_result,"times":time_result }
    
    for j in range(100):
        
        print j
        
        ag = StaticHeuristicApproximatedSarsaLambdaAgent.StaticHeuristicApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[10], my_config=config_heur)
   
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
        total_result.append(res)
        time_result.append(times)
        dict_res["series"]=total_result
        dict_res["times"]=time_result
        with open("Maze_simulation_true_online_static_pheromone_heuristic.pkl", 'wb') as f:
            pickle.dump(dict_res, f)










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
    time_result= []
    dict_res = {"series" : total_result,"times":time_result }
   
    for j in range(100):
        print j
   
        ag = HAApproximatedSarsaLambdaAgent.HAApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[10], my_config=config_heur)

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
        total_result.append(res)
        time_result.append(times)
        dict_res["series"]=total_result
        dict_res["times"]=time_result
        with open("Maze_simulation_true_online_heuristics_hard_model_based_final.pkl", 'wb') as f:
            pickle.dump(dict_res, f)
            
def test3():
    config_heur = {  "Strategy" : "TrueOnline",
                                      "Pheromone_strategy": "hard",
                                      "decrease_exploration" : False, #Mountain Car has a decaying eploration
                                      "learning_rate" : 0.1,
                                      "model_based":False,
                                      "psi": 0.9,
                                      "rho": 0.99,
                                      "eps": 0.1,
                                      "nu":10,            # Epsilon in epsilon greedy policies
                                      "lambda":0.9,
                                      "discount": 0.99,
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
   
        ag = HAApproximatedSarsaLambdaAgent.HAApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[10], my_config=config_heur)
    
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
        total_result.append(res)
        time_result.append(times)
        dict_res["series"]=total_result
        dict_res["times"]=time_result
        with open("Indep_test.pkl", 'wb') as f:
            pickle.dump(dict_res, f)
               


def test4():
 

    config_heur = {  "Strategy" : "TrueOnline",
                                  "Pheromone_strategy": "hard",
                                  "Plan_strategy": "hard",
                                  "decrease_exploration" : False, #Mountain Car has a decaying eploration
                                  "learning_rate" : 0.1,
                                  "psi": 0.3,
                                  "heuristic_dynamic":True,
                                  "model": dyna_model(),
                                  "model_based":True,
                                  "rho": 0.99,
                                  "eps": 0.1,
                                  "nu":10,            # Epsilon in epsilon greedy policies
                                  "n_stop":0, #zero would be a normal Potential Based RS
                                  "lambda":0.9,
                                  "discount": 0.99,
                                  "n_iter": env._max_episode_steps}  
    
    
    total_result = []
    time_result= []
    dict_res = {"series" : total_result,"times":time_result }
    
    for j in range(100):
        print j
   
        ag = APHARSSarsaLambdaAgent.APHARSSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[10], my_config=config_heur)
    
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
        total_result.append(res)
        time_result.append(times)
        dict_res["series"]=total_result
        dict_res["times"]=time_result
        with open("Indep_test.pkl", 'wb') as f:
            pickle.dump(dict_res, f)





def test_dump_trace():
    config_heur = {  "Strategy" : "TrueOnline",
                                      "Pheromone_strategy": "hard",
                                      "decrease_exploration" : False, #Mountain Car has a decaying eploration
                                      "learning_rate" : 0.1,
                                      "model_based":False,
                                      "model":dyna_model(),
                                      "psi": 0.01,
                                      "rho": 0.99,
                                      "eps": 0.1,
                                      "nu":1,            # Epsilon in epsilon greedy policies
                                      "lambda":0.9,
                                      "discount": 0.9,
                                      "n_iter": env._max_episode_steps} 
    
    # rho, psi, nu, hard
    # [0.3, 1e-05, 10]
    # rho, psi, nu, soft
    #[0.9, 1e-05, 5]
    
    total_result = []
    
        #Change the agent here.
    ag = HAApproximatedSarsaLambdaAgent.HAApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[10], my_config=config_heur)
    res = []
    for i in range(episodes):
        rend = False
        
        #if i==50:
        #    rend = True
        
        ag.learn(env,rend)
        res= ag.return_last_steps()
        print res[-1]
        
        if i==199:
            ag.dump_pher_trace("DynaPher.pkl","DynaTC.pkl")
        
    total_result.append(res)
    
    


#test_dump_trace()
test4()
#test2()
#test_dump_trace()
#test_heuristic()