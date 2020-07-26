import numpy as np
from ast import literal_eval
import maze_env


class DoubleQLearning():
    def __init__(self, actions,
                 learning_rate=0.01,
                 discount_rate=0.9,
                 epsilon=0.1,
                 debug=True,):
        self.display_name = "Double Q Learning"
        self.actions = actions
        self.lr = learning_rate
        self.dr = discount_rate
        self.q1 = {}
        self.q2 = {}
        self.debug = debug
        self.epsilon = epsilon

    def init_q_tables(self, observation):
        self.q1.setdefault(observation, {})
        for action in self.actions:
            self.q1[observation].setdefault(action, 0)

        self.q2.setdefault(observation, {})
        for action in self.actions:
            self.q2[observation].setdefault(action, 0)

    def choose_action(self, observation):
        self.init_q_tables(observation)
        if np.random.uniform() <= (1-self.epsilon):  # Greedy
            return (max(self.q1[observation].items(), key=lambda x: x[1]))[0]
        else:  # Epsilon Greedy
            return np.random.choice(4)

    def learn(self, s, a, r, s_):
        a_ = self.choose_action(s_)
        if np.random.uniform() <= 0.5:
            greedy_a = (max(self.q2[s_].items(), key=lambda x: x[1]))[0]
            new_q = self.q2[s_][greedy_a]
            self.q1[s][a] = self.q1[s][a] + self.lr * \
                (r + self.dr*new_q - self.q1[s][a])
        else:
            greedy_a = (max(self.q1[s_].items(), key=lambda x: x[1]))[0]
            new_q = self.q1[s_][greedy_a]
            self.q2[s][a] = self.q2[s][a] + self.lr * \
                (r + self.dr*new_q - self.q2[s][a])
        return s_, a_


if __name__ == "__main__":
    agentXY = [0, 0]
    goalXY = [4, 4]

    wall_shape = np.array([[2, 2], [3, 6]])
    pits = np.array([[6, 3], [1, 4]])
    env = maze_env.Maze(agentXY, goalXY, wall_shape, pits)
    rl_algo = DoubleQLearning(list(range(env.n_actions)))
