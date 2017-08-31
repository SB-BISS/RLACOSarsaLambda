import gym
from gym import envs
from agents import TabularSarsaAgent
from agents import ApproximatedSarsaLambdaAgent
from agents import HAApproximatedSarsaLambdaAgent
import numpy as np
import matplotlib.pyplot as plt
from gym_maze.envs.maze_env import *


print(envs.registry.all())

env =  gym.make("maze-v0")

env._max_episode_steps = 2000

env.reset()

obs_mins = env.observation_space.low
obs_maxs = env.observation_space.high #[env.observation_space[0].max_value, env.observation_space[1].max_value]
print obs_mins
print obs_maxs
discretizations = [5,5]

#Tabular Sarsa + Replacing Elegibility traces
#ag = TabularSarsaAgent.TabularSarsaAgent(discretize(),env.action_space)
#Tile coding Sarsa + Replacing Elegibility Traces
config = {  "Strategy" : "Replacing",
            "decrease_exploration" : False,
            "learning_rate" : 0.1,
            "eps": 0.1,            # Epsilon in epsilon greedy policies
            "lambda":0.9,
            "discount": 0.9,
            "n_iter": 2000} 

ag = ApproximatedSarsaLambdaAgent.ApproximatedSarsaLambdaAgent(obs_mins,obs_maxs,env.action_space,discretizations,[1], my_config=config)


for i in range(1000):
    rend = False
    ag.learn(env,rend)
    res= ag.return_last_steps()
    if i%2 == 0:
        #env.render()
        plt.plot(res)
        plt.draw()
        plt.pause(0.0001)

 #   env.step(env.action_space.sample()) # take a random action