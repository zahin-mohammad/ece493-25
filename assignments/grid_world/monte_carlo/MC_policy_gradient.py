
from MC_brain import MonteCarloAlgorithm
import tensorflow as tf
from ast import literal_eval
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import Dense


class PolicyGradientLearning(MonteCarloAlgorithm):
    def __init__(self, actions,
                num_features=100,  # number of grids
                learning_rate=0.001,
                discount_rate=1,
                debug=True):

        self.display_name = "Policy Gradient Reinforce"
        self.num_actions = len(actions)  # actions 0-3
        self.num_features = num_features
        self.lr = learning_rate
        self.dr = discount_rate
        self.debug = debug
        # self.build_neural_network()



    def discounted_rewards(self, rewards):
        # Calculate discounted rewards, going backwards from end
        discounted_rewards = [0]*len(rewards)
        G = 0
        for i, r in reversed(list(enumerate(rewards))):
            # Gₜ = Rₜ₊₁ + γ * Gₜ₊₁
            G = r + self.dr * G
            discounted_rewards[i] = G

        discounted_rewards = np.array(discounted_rewards)
        # Stabilize the rewards
        # Source: https://datascience.stackexchange.com/questions/20098/why-do-we-normalize-the-discounted-rewards-when-doing-policy-gradient-reinforcem
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards

    def choose_action(self, observation):
        action_probs = self.model.predict(
            self.features(observation)).flatten()
        return np.random.choice(self.num_actions, p=action_probs)

    def learn(self, s, a, r, s_):
        return s_, self.choose_action(s_)

    def features(self, state, action):
        state = literal_eval(state)
        x = int((state[0]-5.0)//40)
        y = int((state[1]-5)//40)
        # One Hot Encoding
        features = np.zeros(self.num_actions*self.num_actions)
        features[(y*10 + x) + self.num_actions*action] = 1
        return features

    def update(self, states, actions, rewards):
        pass

    # Neural Net
    # def update(self, states, actions, rewards):
    #     y = self.discounted_rewards(rewards)

    #     x = np.array([self.features(state)
    #                   for state, action in zip(states, actions)])
    #     self.model.fit(x, y, epochs=1000, batch_size=1000, verbose=0)
    #     _, accuracy = self.model.evaluate(x, y, verbose=1)
    #     print('Accuracy: %.2f' % (accuracy*100))

    # def build_neural_network(self):
    #     self.model = keras.Sequential()

    #     self.model.add(
    #         Dense(50, input_dim=2, activation=tf.nn.tanh))
    #     self.model.add(Dense(4, activation=tf.nn.softmax))

    #     opt = keras.optimizers.SGD(learning_rate=self.lr)
    #     self.model.compile(loss='mean_squared_error',
    #                        optimizer=opt, metrics=['accuracy'])
