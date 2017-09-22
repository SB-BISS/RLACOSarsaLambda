import pickle
import numpy as np
from gym_maze.envs.maze_env import *
from gym_maze.envs.dyna_maze import Maze
from tilecoding.representation import TileCoding
 
    #Draw the pheromone map previously saved
pheromone_map = np.zeros((30, 30))
with open("pher_trace_10.pkl", "rb") as fh:
    trace = pickle.load(fh)
    
with open("tile_coder_dyna.pkl", "rb") as fh2:
    tc = pickle.load(fh2)

print tc

for x in range(0, 30):
    for y in range(0, 30):
        for a in range(0, 4):
            indexes = tc[a].__call__(np.array([x, y]))
            value = np.sum(trace[a][indexes])
            pheromone_map[x, y] += value 
    
max_pher = max(pheromone_map.flatten())
pheromone_map = pheromone_map / max_pher

give_me_a_maze = Maze() # standard maze used for experimentation

while True:
    give_me_a_maze._render_pheromone(pheromone_map)

