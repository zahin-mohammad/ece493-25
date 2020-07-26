
import numpy as np
import json
import sys
from DP_brain import DynamicProgrammingAlgorithm
from util import *
from ast import literal_eval


class PolicyIteration(DynamicProgrammingAlgorithm):
    def __init__(self, actions, environment, theta=0.001, discount_rate=0.9, debug=False):
        DynamicProgrammingAlgorithm.__init__(self, actions, environment)
        self.display_name = "Policy Iteration"
        self.debug = debug
        self.theta = theta
        self.discount_rate = discount_rate

        self.initialize_value()
        self.initialize_policy()
        self.policy_improvement()

        if self.debug:
            with open('PI_action_values.json', 'w') as fp:
                json.dump(to_json(self.values), fp)
            with open('PI_policy.json', 'w') as fp:
                json.dump(to_json(self.policy), fp)

    def choose_action(self, observation):
        observation = tuple(literal_eval(observation))
        return self.policy[observation]

    def learn(self, s, a, r, s_):
        return s_, self.choose_action(s_)

    def initialize_value(self):
        for state in self.env.keys():
            self.values[state] = 0

    def initialize_policy(self):
        for state in self.env.keys():
            self.policy[state] = np.random.choice(list(range(3)))

    def policy_evaluation(self):
        while True:
            delta = 0
            for state in self.env.keys():
                v = self.values[state]
                self.values[state] = 0
                next_state, reward, _ = self.env[state][self.policy[state]]
                self.values[state] += (reward +
                                       self.discount_rate*self.values[next_state])

                delta = max(delta, abs(v-self.values[state]))
            if delta < self.theta:
                break
        self.policy_improvement()

    def iterative_value_calc(self, state):
        actions = {}
        for action in self.actions:
            next_state, reward, _ = self.env[state][action]
            actions[action] = (reward + self.discount_rate *
                               self.values[next_state])
        return actions

    def policy_improvement(self):
        policy_stable = True
        for state in self.env.keys():
            old_action = self.policy[state]

            arg_max_a, _ = max(self.iterative_value_calc(
                state).items(), key=lambda x: x[1])
            self.policy[state] = arg_max_a

            if old_action != arg_max_a:
                policy_stable = False
        if policy_stable:
            return
        else:
            self.policy_evaluation()
        return
