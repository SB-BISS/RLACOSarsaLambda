
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten,Input
from keras.optimizers import Adam
from keras.models import Model
#Problem number 1: the memory.
# The memory should not be discrete, it should be a distribution.
# The distribution should modify itself slowly as new points come in




# Deep Dynamic Heuristically Accelerated Q-learning Agent
class ACODQNAgent:
    def __init__(self, state_size, action_size,config):
        self.nodes = config["neurons"]
        self.best_trajectory= ([],0)
        self.cum_reward = 0
        self.active_learn = config["active_learn"]
        self.episode = 0
        self.psi = config["psi"]
        self.nu =config["nu"]
        self.iter = config["n_iter"]
        self.pheromone_decay = 0.99
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=config["buffer"])
        self.memory_aco = deque(maxlen=config["buffer"]) #these are tajectories to remember...
        self.gamma = config["discount"]    # discount rate
        self.lmbd = 0.9
        self.warmup = 100
        self.epsilon =config["eps"]  # exploration rate
        self.epsilon_min = 0.01
        self.rep = config["batch"]
        self.epsilon_decay = config["eps_decay"]
        self.learning_rate = config["learning_rate"]
        self.model = self._build_model()
        self.model_aco = self._build_dynamic_aco_heuristic()
        self.heur_activate= False

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        #model = Sequential()
        omega = Input(shape=(self.state_size,))
        f = Dense(self.nodes, activation='relu')(omega)
        g = Dense(self.nodes, activation='relu')(f)
        h = Dense(self.action_size, activation='linear')(g)
        model = Model(inputs= omega, outputs = h)
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))

        print model.summary()

        return model

    def _build_dynamic_aco_heuristic(self):
        # Neural Net for ACO
        return self._build_model()



    def remember_transition(self,state, action, quality, next_state, done):
        #it should be the case that the value of the trajectory is saved
        #pass
        self.memory_aco.append((state, action, quality, next_state, done))


    #getting the trajectory probably
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        #if self.gmix!=None:
            #vector = np.append(np.append(np.append(state,action),reward),next_state)
            #gmm_map_qb(vector,self.gmix)

        #print len(self.memory)


    def act_normal(self,state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        #print self.model.get_input_shape_at(0)

        #state.shape = (self.state_size,1)

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def act(self, state):
        rand_val = np.random.rand()
        if  rand_val<= self.epsilon:
            action = random.randrange(self.action_size)
            #print action
            return action
        #state.shape = (self.state_size, 1)
        act_values = self.model.predict(state)
        act_values_h = self.model_aco.predict(state) # give me the ACO values

        arg_h=np.argmax(act_values_h[0])

        current_best=  np.argmax(act_values[0])
        best_sa_value= act_values[0][current_best]
        challenging_best =  act_values[0][arg_h] +  self.psi*(best_sa_value-  act_values[0][arg_h] +self.nu)

        if challenging_best > act_values[0][current_best]:
            #print (challenging_best,best_sa_value)
            return arg_h
        else:
            return current_best

    #decay in batches. since we have weights, the NN will adapt consequently
    def decay_heuristic(self):
        minibatch = random.sample(self.memory_aco, self.rep)
        for state, action, quality, next_state, done in minibatch:

            target_f = self.model_aco.predict(state)
            target_f[0][action] = target_f[0][action]*self.pheromone_decay
            self.model_aco.fit(state, target_f, epochs=1, verbose=0)

    def train_one_step(self,state,action,reward,next_state,done):
        target = reward
        if not done:



            target = reward + self.gamma * \
                              np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(state)
        #print target_f
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

    def replay_aco(self,batch_size):
            minibatch = random.sample(self.memory_aco, batch_size)
            for state, action, quality, next_state, done in minibatch:
                target = quality
                target_f = self.model_aco.predict(state)
                target_f[0][action] = target + self.pheromone_decay*target_f[0][action]
                self.model_aco.fit(state, target_f, epochs=1, verbose=0)

    def replay_best(self):
        minibatch= random.sample(self.best_trajectory[0],np.min([len(self.best_trajectory),self.rep]))
        for state, action,_, next_state, done in minibatch:
            target = self.best_trajectory[1]
            target_f = self.model_aco.predict(state)
            target_f[0][action] = target + self.pheromone_decay * target_f[0][action]
            self.model_aco.fit(state, target_f, epochs=1, verbose=0)

    def replay(self, batch_size):
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in minibatch:


                target = reward
                done = False
                if not done:
                    target = reward + self.gamma * \
                                      np.amax(self.model.predict(next_state)[0])
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay



    def learn(self,env, rend = False):
        self.episode = self.episode +1
        s = env.reset()
        s = np.reshape(s, [1, self.state_size])
        self.cum_reward = 0
        current_trajectory = []

        for t in range(self.iter):

            if self.heur_activate or self.active_learn==False:
                action = self.act(s)
            else:
                action = self.act_normal(s)

            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            next_state, reward, done, _ = env.step(action)
            self.cum_reward = self.cum_reward +reward
            next_state = np.reshape(next_state, [1,self.state_size])
            # Remember the previous state, action, reward, and done
            self.remember(s, action, reward, next_state, done)
            current_trajectory.append((s, action, reward, next_state, done))
            #if self.warmup>self.episode:
            if self.active_learn:
                self.train_one_step(s,action,reward,next_state,done)

            trace_rate = self.learning_rate

            # make next_state the new current state for the next frame.
            s = next_state
            if rend:
                env.render()
            if done:  #
                break

        quality = 0
        if self.cum_reward < 0:
            quality = np.min([0.99, 1.0 / np.abs(self.cum_reward)])
        else:
            partial = self.cum_reward
            if  partial<1:
                partial=1
            quality = 2 - 1 / (partial)

        if quality> self.best_trajectory[1]:
                self.heur_activate=True
                print (self.best_trajectory[1],quality)
                self.best_trajectory = (current_trajectory,quality)

        for i  in range(len(current_trajectory)): # variability
               s,a,rw, ns,done = current_trajectory[i]
               self.remember_transition(s,a,quality,ns,done)


        #train the agent with the experience of the episode
        #if (self.rep<len(self.memory)):
        #    self.replay(self.rep)
        #else:
        #    self.replay(len(self.memory)-1)

        #do this only if there is something in the heuristic side
        if(self.rep<len(self.memory_aco) and self.active_learn):
                self.replay_best()
                self.replay_aco(self.rep)
                self.decay_heuristic()

    def return_cum_reward(self):
        return self.cum_reward

    def save_models(self,filename_main,filename_heur):
        self.model.save_weights(filename_main)
        self.model_aco.save_weights(filename_heur)

    def load_models(self,filename_main,filename_heur):
        self.model.load_weights(filename_main)
        self.model_aco.load_weights(filename_heur)