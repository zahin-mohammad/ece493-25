import numpy as np
from ast import literal_eval
import maze_env


class ExpectedSarsaLearning():
    def __init__(self, actions,
                 learning_rate=0.015,
                 discount_rate=0.9,
                 epsilon=0.1):
        self.display_name = "Expected Sarsa Learning"
        self.actions = actions
        self.lr = learning_rate
        self.dr = discount_rate
        self.q = {}
        self.epsilon = epsilon

    def choose_action(self, observation):
        # Init Q table
        self.q.setdefault(observation, {})
        for action in self.actions:
            self.q[observation].setdefault(action, 0)
        if np.random.uniform() <= (1-self.epsilon):  # Greedy
            return (max(self.q[observation].items(), key=lambda x: x[1]))[0]
        else:  # Epsilon Greedy
            return np.random.choice(4)

    def learn(self, s, a, r, s_):
        a_ = self.choose_action(s_)

        state_action_values = np.array([self.q[s_][action]
                                        for action in self.actions])
        # weighted probability
        probabilities = state_action_values / np.sum(state_action_values)

        expected_q = np.sum(probabilities * np.array([self.q[s_][action]
                                                      for action in self.actions]))
        # expected_q = sum([self.q[s_][action]
        #   for action in self.actions])
        self.q[s][a] = self.q[s][a] + \
            self.lr*(r + self.dr*expected_q-self.q[s][a])
        return s_, a_


if __name__ == "__main__":
    agentXY = [0, 0]
    goalXY = [4, 4]

    wall_shape = np.array([[2, 2], [3, 6]])
    pits = np.array([[6, 3], [1, 4]])
    env = maze_env.Maze(agentXY, goalXY, wall_shape, pits)
    rl_algo = ExpectedSarsaLearning(list(range(env.n_actions)))
