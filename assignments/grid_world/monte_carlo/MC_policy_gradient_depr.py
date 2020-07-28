
from MC_brain import MonteCarloAlgorithm
import tensorflow as tf
from ast import literal_eval
from tensorflow import keras
import numpy as np

class PolicyGradientLearning(MonteCarloAlgorithm):
    def __init__(self, actions, learning_rate=0.01, discount_rate=0.9, epsilon = 0.1, debug=True,):
        self.display_name="Policy Gradient"
        self.actions = actions
        self.lr = learning_rate
        self.dr = discount_rate

        self.neural_network = keras.Sequential([
            keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
            keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
            keras.layers.Dense(len(self.actions), activation='softmax')
        ])
        self.neural_network.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=self.lr))

        self.debug = debug
        self.epsilon = epsilon

    def discounted_rewards(self, rewards):
        # Calculate discounted rewards, going backwards from end
        discounted_rewards = []
        G = 0
        for r in reversed(rewards):
            # G_t = R_{t+1} + \gamma * G_{t+1} 
            G = r + self.dr * G
            discounted_rewards.insert(0, G)
        return np.array(discounted_rewards)
    
    def choose_action(self, observation):
        observation = list(literal_eval(observation))

        softmax_out = self.neural_network(np.array([observation]).reshape((1, -1)))

        selected_action = np.random.choice(len(self.actions), p=softmax_out.numpy()[0])
        print(selected_action)
        return selected_action              

    def learn(self, s, a, r, s_):
        return s_, self.choose_action(s)
    
    def update(self, states, actions, rewards):
        discounted_rewards = self.discounted_rewards(rewards)
        # standardise the rewards
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        # col 
        # for i, s in enumerate(states):
        #     state = list(literal_eval(s))
        #     print(state)
        #     x = int((state[0]-5.0)/40.0)
        #     y = int((state[1]-5.0)/40.0)
        #     states[i] = y*10 + x
        # for s in states:
        #     print(s)
        states = np.vstack([np.array(literal_eval(state), dtype=np.float32) for state in states])
        loss = self.neural_network.train_on_batch(states, np.array([np.array([reward]*len(self.actions)) for reward in discounted_rewards]))
        return loss
    