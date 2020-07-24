import numpy as np
from ast import literal_eval
import json
import maze_env
from util import *
from RL_brain import RLAlgorithm

class ValueIteration(RLAlgorithm):
    def __init__(self, actions, environment, theta=0.001, discount_rate=0.9, debug=False):
        RLAlgorithm.__init__(self, actions, environment)
        self.display_name="Value Iteration"
        self.debug = debug
        self.discount_rate = discount_rate
        self.create_value_policy_function(theta)

    def init_state_values(self):
        for state in self.env.keys():
            self.values[state] = 0

    # Return estimated expected rewards from current state
    def iterative_value_calc(self, o_state):
        action_values = {}
        for action in self.actions:
            next_state, reward, _ = self.env[o_state][action]
            action_values[action] = reward + self.discount_rate*self.values[next_state]

        return action_values
    
    # Creates Value Function V(S) and optimal policy Pi(S)
    def create_value_policy_function(self, theta):
        self.init_state_values()
        counter = 0

        while True:
            delta = 0
            for state in self.env.keys():
                actions = self.iterative_value_calc(state).items()
                best_action, best_action_value = max(actions, key=lambda x: x[1])
            
                delta = max(delta, abs(best_action_value - self.values[state])) 
                
                self.values[state] = best_action_value
                self.policy[state] = best_action

            if delta < theta:
                break
            counter +=1
        if self.debug:
            print(f'Sweeps: {counter}')
            with open('VI_action_values.json', 'w') as fp:
                json.dump(to_json(self.values), fp)
            with open('VI_policy.json', 'w') as fp:
                json.dump(to_json(self.policy), fp)

    def choose_action(self, observation):
        observation =  tuple(literal_eval(observation))
        return self.policy[observation]
    
    def learn(self, s, a, r, s_):
        return s_, self.choose_action(s_)

if __name__ == "__main__":
    agentXY=[0,0]
    goalXY=[4,4]

    wall_shape = np.array([[2,2],[3,6]])
    pits = np.array([[6,3],[1,4]])
    env = maze_env.Maze(agentXY,goalXY,wall_shape,pits)
    rl_algo = ValueIteration(list(range(env.n_actions)), env)