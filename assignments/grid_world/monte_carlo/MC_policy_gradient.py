
from MC_brain import MonteCarloAlgorithm
import tensorflow as tf
from ast import literal_eval
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import Dense

class PolicyGradientLearning(MonteCarloAlgorithm):
    def __init__(self, actions, num_features,
        learning_rate=0.01, 
        discount_rate=0.9, 
        epsilon = 0.1,
        debug=True):

        self.display_name="Policy Gradient Reinforce"
        self.num_actions = len(actions) # actions 0-3
        self.num_features = num_features
        self.lr = learning_rate
        self.dr = discount_rate

        self.debug = debug
        self.epsilon = epsilon

        self.build_neural_network()

    def build_neural_network(self):
        self.model = keras.Sequential()
        self.model.add(Dense(50, input_dim=self.num_features, activation='relu'))    
        self.model.add(Dense(25, activation='relu'))
        self.model.add(Dense(self.num_actions, activation='softmax'))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'],)

    def discounted_rewards(self, rewards):
        # Calculate discounted rewards, going backwards from end
        discounted_rewards = []
        G = 0
        for r in reversed(rewards):
            # G_t = R_{t+1} + \gamma * G_{t+1} 
            G = r + self.dr * G
            discounted_rewards.insert(0, G)
        
        discounted_rewards = np.array(discounted_rewards)
        # Stabalize the rewards
        # Source: https://datascience.stackexchange.com/questions/20098/why-do-we-normalize-the-discounted-rewards-when-doing-policy-gradient-reinforcem
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards
    
    def choose_action(self, observation):
        features = self.features(observation)
        action_probabilities = self.model.predict(features)
        return np.random.choice(self.num_actions, p=action_probabilities.flatten())             

    def learn(self, s, a, r, s_):
        return s_, self.choose_action(s)
    
    def features(self, state):
        state_features = np.array([list(literal_eval(state))])
        return state_features

    def update(self, states, actions, rewards):
        y = self.discounted_rewards(rewards)
        x = np.array([np.array(self.features(state)) for state in states])
        self.model.fit(x, y, epochs=150, batch_size=10, verbose=0)
        _, accuracy = self.model.evaluate(x, y)
        print('Accuracy: %.2f' % (accuracy*100))
    