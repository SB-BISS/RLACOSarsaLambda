import gym
from gym import envs
from agents import TabularSarsaAgent
from agents import ApproximatedSarsaLambdaAgent
from agents import HAApproximatedSarsaLambdaAgent
from agents import PAApproximatedSarsaLambdaAgent
from agents import APHARSSarsaLambdaAgent 

from agents import StaticHeuristicApproximatedSarsaLambdaAgent
from agents.DQNAgent import DQNAgent
from agents.ACODQNAgent import ACODQNAgent
from static_heuristics.LoadableHeuristic import LoadableHeuristic #import LoadableHeuristic
import numpy as np
import matplotlib.pyplot as plt
import pickle
from model import mc_model
import time

print(envs.registry.all())

env = gym.make('LunarLander-v2') #Defined by Stefano Bromuri
episodes_running = 10000
env._max_episode_steps = 1000

env.reset()

print env.observation_space
obs_mins = env.observation_space.low
obs_maxs = env.observation_space.high #[env.observation_space[0].max_value, env.observation_space[1].max_value]


#obs_mins[0:8] = -2.0
#obs_maxs[0:8] = 2.0

print obs_mins
print obs_maxs



discretizations = [3,3,3,3,3,4,3,3]
layers = 3

config = {  "Strategy" : "TrueOnline",
            "decrease_exploration" : True,
            "learning_rate" : 0.5,
            "eps": 0.025,            # Epsilon in epsilon greedy policies
            "lambda":0.9,
            "discount": 1,
            "n_iter": env._max_episode_steps} 

config_heur = {  "Strategy" : "Replacing",
                                  "Pheromone_strategy": "hard",
                                  "Plan_strategy": "soft",
                                  "decrease_exploration" : True, #Mountain Car has a decaying eploration
                                  "learning_rate" : 0.01,
                                  "psi": 0.001,
                                  "heuristic_dynamic":True,
                                  "model": mc_model.mc_model(),
                                  "model_based":False,
                                  "rho": 0.99,
                                  "eps": 0.1,
                                  "nu":0.010,            # Epsilon in epsilon greedy policies
                                  "lambda":0.9,
                                  "discount": 1,
                                  "n_stop":3, #Potential steps
                                  "n_iter": env._max_episode_steps}

config_dqn = {"Strategy": "Replacing",
               "decrease_exploration": True,  # Mountain Car has a decaying eploration
               "learning_rate": 0.0005,
               "psi": 0.01,
               "heuristic_dynamic": True,
               "model_based": False,
               "rho": 0.995,
               "eps": 0.05,
               "eps_decay": 0.995,
               "nu": 10,  # Epsilon in epsilon greedy policies
               "lambda": 0.9,
               "neurons":32,
               "discount": 0.95,
               "batch": 32,  # Potential steps
                "buffer": 2000,
               "n_iter": env._max_episode_steps}



total_result = []
time_result= []

dict_res = {"series" : total_result,"times":time_result }

print env.action_space
print env.observation_space

#ag = ApproximatedSarsaLambdaAgent.ApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[layers], my_config=config_heur)
#ag = HAApproximatedSarsaLambdaAgent.HAApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[layers], my_config=config_heur)

ag = ACODQNAgent(len(obs_mins),env.action_space.n,config_dqn)
    
res = []
times = []
for i in range(episodes_running):

    rend = False
    #if i % 100==0:
    #    rend = True

    time_before = time.time()
    ag.learn(env,rend)
    rew= ag.return_cum_reward()
    res = np.append(res,rew)
    time_after = time.time()
    time_req = time_after - time_before
    times.append(time_req)

    ag.save_models("qmodel_lunar.hp5","heurmodel_lunar.hp5")

    if i % 100 == 0:
     plt.plot(res)
     plt.draw()
     plt.show(block=False)
     plt.pause(0.0001)

    print res[-1]

