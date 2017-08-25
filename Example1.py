import gym
from gym import envs
from gym_m3d.envs.m3d_environment import m3d
from agents import TabularSarsaAgent
from agents import ApproximatedSarsaLambdaAgent
from agents import HAApproximatedSarsaLambdaAgent
import numpy as np
import matplotlib.pyplot as plt

print(envs.registry.all())

env = gym.make('MountainCar-v0') #Defined by Stefano Bromuri

env._max_episode_steps = 500

env.reset()

obs_mins = env.observation_space.low
obs_maxs = env.observation_space.high #[env.observation_space[0].max_value, env.observation_space[1].max_value]
discretizations = [9,8]

#Tabular Sarsa + Replacing Elegibility traces
#ag = TabularSarsaAgent.TabularSarsaAgent(discretize(),env.action_space)
#Tile coding Sarsa + Replacing Elegibility Traces
config = {  "Strategy" : "Replacing",
            "decrease_exploration" : True,
            "learning_rate" : 0.5,
            "eps": 0.025,            # Epsilon in epsilon greedy policies
            "lambda":0.9,
            "discount": 1,
            "n_iter": 500} 

ag = HAApproximatedSarsaLambdaAgent.HAApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[5], my_config=config)


for i in range(1000):
    rend = False
    ag.learn(env,rend)
    res= ag.return_last_steps()
    if i%10 == 0:
        plt.plot(res)
        plt.draw()
        plt.pause(0.0001)

 #   env.step(env.action_space.sample()) # take a random action