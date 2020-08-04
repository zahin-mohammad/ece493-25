
from MC_brain import MonteCarloAlgorithm
from ast import literal_eval
import numpy as np
from math import exp

np.random.seed(10)

class PolicyGradientLearning(MonteCarloAlgorithm):
    def __init__(self, actions,
                num_features=100,  # number of grids
                learning_rate=0.015,
                discount_rate=0.99,
                debug=True):
        self.display_name = "Policy Gradient Reinforce"
        self.num_actions = len(actions)  # actions 0-3
        self.num_features = num_features
        self.theta = np.zeros(self.num_features*self.num_actions)

        self.lr = learning_rate
        self.dr = discount_rate
        self.debug = debug



    def discounted_rewards(self, rewards):
        # Calculate discounted rewards, going backwards from end
        discounted_rewards = [0]*len(rewards)
        G = 0
        for i, r in reversed(list(enumerate(rewards))):
            # Gₜ = Rₜ₊₁ + γ * Gₜ₊₁
            G = r + self.dr * G
            discounted_rewards[i] = G
        return np.array(discounted_rewards)

    def choose_action(self, observation):
        action_prob = [ self.policy(a, observation) for a in range(self.num_actions) ]
        return np.random.choice(range(self.num_actions), p = action_prob)

    def learn(self, s, a, r, s_):
        return s_, self.choose_action(s_)

    def features(self, state, action):
        state = literal_eval(state)
        x = int((state[0]-5.0)//40)
        y = int((state[1]-5)//40)
        # One Hot Encoding
        features = np.zeros(self.num_features*self.num_actions)
        features[(y*10 + x) + self.num_features*action] = 1
        return features

    def update(self, states, actions, rewards):
        discounted_rewards = self.discounted_rewards(rewards)
        t = 0
        for state, action, reward in zip(states,actions,discounted_rewards):
            grad_ln_pi = self.features(state, action) - sum([self.policy(b,state)*self.features(state,b) for b in range(self.num_actions)])
            self.theta = self.theta + self.lr*(self.dr ** t)*reward * grad_ln_pi
            t += 1
        # self.dr = min(1, self.dr + 0.00002)

    def policy(self, action, observation):
        def h(a):
            return exp(self.theta @ self.features(observation,a))
        out = h(action) / sum([h(a) for a in range(self.num_actions)])
        return out