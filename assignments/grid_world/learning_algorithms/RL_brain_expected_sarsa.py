import numpy as np
from ast import literal_eval
import pandas as pd


class ExpectedSarsaLearning():
    def __init__(self, actions,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 epsilon=0.1):
        self.display_name = "Expected Sarsa Learning"
        self.actions = actions  
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = epsilon
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() >= self.epsilon:  # Greedy
            return self.q_table.loc[observation, :].argmax() 
        else:  # Epsilon Greedy
            return np.random.choice(self.actions)

    def learn(self, s, a, r, s_):
        a_ = self.choose_action(s_)

        state_action_values = self.q_table.loc[s_,:]
        max_value = np.max(state_action_values)
        max_count = len(state_action_values[state_action_values == max_value])
        
        # All actions get eps/num_actions prob
        # Best actions get 1- eps (the remainder)
        expected_value_max =  ((1 - self.epsilon) / max_count + self.epsilon / len(self.actions)) * max_count * max_value
        expected_value_non_max = (np.sum(state_action_values) - max_value * max_count) * (self.epsilon / len(self.actions))

        self.q_table.loc[s, a] = self.q_table.loc[s, a] + self.lr * (r+ self.gamma * (expected_value_max + expected_value_non_max) - self.q_table.loc[s, a])
        return s_, a_

    '''States are dynamically added to the Q(S,A) table as they are encountered'''
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )