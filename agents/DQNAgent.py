
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam



# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size,config):
        self.cum_reward = 0
        self.iter = config["n_iter"]
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.rep = 128
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        #print len(self.memory)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self,env, rend = False):

        s = env.reset()
        s = np.reshape(s, [1, self.state_size])
        self.cum_reward = 0
        for t in range(self.iter):
            action = self.act(s)
            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            next_state, reward, done, _ = env.step(action)
            self.cum_reward = self.cum_reward +reward
            next_state = np.reshape(next_state, [1,self.state_size])
            # Remember the previous state, action, reward, and done
            self.remember(s, action, reward, next_state, done)
            # make next_state the new current state for the next frame.
            s = next_state
            if rend:
                env.render()
            if done:  #
                break

        # train the agent with the experience of the episode
        if (self.rep<len(self.memory)):
            self.replay(self.rep)
        else:
            self.replay(len(self.memory)-1)

    def return_cum_reward(self):
        return self.cum_reward