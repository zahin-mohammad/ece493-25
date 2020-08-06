import numpy as np
from ast import literal_eval
import collections
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import Adam
import random

np.random.seed(10)
random.seed(10)

class DeepQNetwork():
    def __init__(self, actions,
        num_features = 4, # x1 y1 x2 y2
        batch_size = 32,
        memory_capacity = 500,
        learning_rate=0.01, 
        discount_rate=0.9,
        epsilon_initial = 1.0,
        epsilon_min = 0.1, 
        epsilon_decay = 0.9999):

        self.actions = actions
        self.num_features = num_features
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.memory = collections.deque() 

        self.lr = learning_rate
        self.dr = discount_rate
        
        self.epsilon = epsilon_initial
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.model = self.__build_model__()
        self.target_model = self.__build_model__()
        self.iteration_counter = 0
        self.replace_target_model = 300
        
        self.display_name="Double Deep Q Network"


    def choose_action(self, observation):
        s = self.__get_features__(observation)
        if np.random.uniform() <= self.epsilon:
            return np.random.choice(self.actions)
        return np.argmax(self.model.predict(s)[0])
                

    def learn(self, observation_s, a, r, observation_s_):
        self.__remember__(observation_s,a,r,observation_s_)
        self.__train_model__()
        self.__train_target_model__()
        a_ = self.choose_action(observation_s_)
        
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
        print(self.epsilon)
        self.iteration_counter += 1
        return observation_s_, a_

    def __build_model__(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.num_features, 
            activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(len(self.actions)))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.lr))
        return model


    def __remember__(self, observation_s, a, r, observation_s_):
        isDone = observation_s_ == 'terminal'
        s, s_ = self.__get_features__(observation_s), self.__get_features__(observation_s_)
        
        self.memory.append([s,a,r,s_, isDone])
        if len(self.memory) > self.memory_capacity:
            self.memory.popleft()


    def __get_features__(self, observation):   
        return np.array([literal_eval(observation)])

    def __train_model__(self):
        if len(self.memory) < self.batch_size: 
            return
        samples = random.sample(self.memory, self.batch_size)
        x = []
        y = []
        for s, a, r, s_, isDone in samples:
            target = self.target_model.predict(s)
            target[0][a] = r if isDone else (r + self.dr*max(self.target_model.predict(s_)[0]))
            x.append(s)
            y.append(target[0])
        x = np.vstack(x)
        y = np.array(y)
        self.model.fit(x, y, epochs=1, verbose=0)
    
    def __train_target_model__(self):
        # can i just directly pass weights?
        if self.iteration_counter % self.replace_target_model:
            return
        weights = self.model.get_weights()
        self.target_model.set_weights([weights[i] for i in range(len(self.target_model.get_weights()))])        



