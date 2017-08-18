import gym
from gym import envs
from gym_m3d.envs.m3d_environment import m3d
from agents import TabularSarsaAgent
import numpy as np
import matplotlib.pyplot as plt

print(envs.registry.all())

env = gym.make('MountainCar-v0') #Defined by Stefano Bromuri

env._max_episode_steps = 500
print(env.action_space)# actions in the MC problem
#> Discrete(2)
print(env.observation_space)#
 
 
def discretize():    
        # state discretization for the mountain car problem
        xdiv  = (0.6-(-1.2))   / 10.0
        xpdiv = (0.07-(-0.07)) / 5.0
        
        x = np.arange(-1.5,0.5+xdiv,xdiv)
        xp= np.arange(-0.07,0.07+xpdiv,xpdiv)

        N=np.size(x)
        M=np.size(xp)

        states=[] #zeros((N*M,2)).astype(Float32)
        index=0
        for i in range(N):    
            for j in range(M):
                states.append([x[i], xp[j]])
                
        return np.array(states)


 
 

env.reset()


ag = TabularSarsaAgent.TabularSarsaAgent(discretize(),env.action_space)

for i in range(1000):
    rend = False
    ag.learn(env,rend)
    res= ag.return_last_steps()
    if i%100 == 0:
        plt.plot(res)
        plt.show()
 #   env.step(env.action_space.sample()) # take a random action