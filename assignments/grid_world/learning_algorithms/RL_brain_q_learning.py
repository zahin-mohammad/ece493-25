import numpy as np
from ast import literal_eval

class QLearning():
    def __init__(self, actions, learning_rate=0.01, discount_rate=0.9, epsilon = 0.1, debug=True,):
        self.display_name="Q Learning"
        self.actions = actions
        self.lr = learning_rate
        self.dr = discount_rate
        self.q = {}
        self.debug = debug
        self.epsilon = epsilon
 
    def lazy_table_init(self, observation):
        self.q[observation] = {action:0 for action in self.actions} if observation not in self.q else self.q[observation]

    def choose_action(self, observation):
        self.lazy_table_init(observation)
        if np.random.uniform() >= self.epsilon: # Greedy
            return (max(self.q[observation].items(), key=lambda x: x[1]))[0]
        else: # Epsilon Greedy
            return np.random.choice(self.actions)
                

    def learn(self, s, a, r, s_):
        a_ = self.choose_action(s_)            
        greedy_a = (max(self.q[s_].items(), key=lambda x: x[1]))[0]
        new_q = self.q[s_][greedy_a]
        self.q[s][a] = self.q[s][a] + self.lr*(r + self.dr*new_q - self.q[s][a])
        return s_, a_
