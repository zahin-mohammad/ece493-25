import numpy as np
from ast import literal_eval
import maze_env


class EligibilityTraceSarsaLearning():
    def __init__(self, actions,
                 learning_rate=0.01,
                 discount_rate=0.9,
                 decay_rate=0.9,
                 epsilon=0.1):

        self.actions = actions
        self.discount_r = discount_rate
        self.decay_r = decay_rate
        self.learning_r = learning_rate
        self.epsilon = epsilon

        self.display_name = "Eligibility Trace Sarsa Learning"
        self.q = {}  # Q table
        self.e = {}  # Eligibility Trace

    def lazy_table_init(self, observation):
        self.q[observation] = {action:0 for action in self.actions} if observation not in self.q else self.q[observation]
        self.e[observation] = {action:0 for action in self.actions} if observation not in self.e else self.e[observation]

    def choose_action(self, observation):
        self.lazy_table_init(observation)

        if np.random.uniform() <= (1-self.epsilon):  # Greedy
            return (max(self.q[observation].keys(), key=lambda x: self.q[observation][x]))
        else:  # Epsilon Greedy
            return np.random.choice(self.actions)

    def learn(self, s, a, r, s_):
        a_ = self.choose_action(s_)
        td_error = r + self.discount_r*self.q[s_][a_] - self.q[s][a]
        self.e[s][a] += 1
        # TODO: Can do this easily via np arrays
        for state in self.q.keys():
            for action in self.q[state]:
                self.q[state][action] += self.learning_r * \
                    td_error*self.e[state][action]
                self.e[state][action] = self.discount_r * \
                    self.decay_r*self.e[state][action]
        return s_, a_
