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

class DoubleDeepQNetwork():
    def __init__(self, actions,
        num_features = 4, # x1 y1 x2 y2
        batch_size = 32,
        memory_capacity = 2000,
        learning_rate=0.001, 
        discount_rate=0.9,
        epsilon_initial = 1.0,
        epsilon_min = 0.001, 
        epsilon_decay = 0.9999):

        self.actions = actions
        self.num_features = num_features
        self.batch_size = batch_size
        self.memory = collections.deque(maxlen=memory_capacity) 
        self.training_threshold = 1000

        self.lr = learning_rate
        self.dr = discount_rate
        
        self.epsilon = epsilon_initial
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.model = self.__build_model__()
        self.target_model = self.__build_model__()
        self.target_model.set_weights(self.model.get_weights())

        self.iteration_counter = 0
        self.replace_target_model = 300
        
        self.display_name="Double Deep Q Network"


    def choose_action(self, observation):
        s = self.__get_features__(observation)
        if np.random.uniform() <= self.epsilon:
            return np.random.choice(self.actions)
        return np.argmax(self.model.predict(s.reshape(len(self.actions),1).T)[0])
                

    def learn(self, observation_s, a, r, observation_s_):
        self.__remember__(observation_s,a,r,observation_s_)
        self.__train_model__(observation_s_)
        self.__train_target_model__()
        a_ = self.choose_action(observation_s_)
        
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
        # if not (self.iteration_counter % 2000):
        #     self.epsilon = min(self.epsilon*2, 1.0)
        self.iteration_counter += 1
        return observation_s_, a_

    def __build_model__(self):
        model = Sequential()
        model.add(Dense(20, input_dim=self.num_features, activation="relu"))
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



    def __get_features__(self, observation):   
        return np.array(literal_eval(observation))

    def __train_model__(self, observation_s_):
        if len(self.memory) < max(self.training_threshold, self.batch_size): 
            return
        samples = random.sample(self.memory, self.batch_size)

        curr_states = np.array([sample[0] for sample in samples])
        curr_qs = self.model.predict(curr_states)
        
        new_curr_states = np.array([sample[3] for sample in samples])
        new_curr_qs = self.target_model.predict(new_curr_states)

        x, y = [], []

        for index, (s, a, r, s_, isDone) in enumerate(samples):
            new_q = r if isDone else (r + self.dr*max(new_curr_qs[index]))
            curr_q = curr_qs[index]
            curr_q[a] = new_q
            
            x.append(s)
            y.append(curr_q)

        self.model.fit(
            np.array(x), 
            np.array(y), 
            batch_size=self.batch_size, 
            verbose=0,
            shuffle=False,
            ) 
    
    def __train_target_model__(self):
        if self.iteration_counter % self.replace_target_model:
            return
        self.target_model.set_weights(self.model.get_weights())        
