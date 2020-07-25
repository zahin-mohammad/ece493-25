import numpy as np
from ast import literal_eval
import maze_env


class EligibilityTraceSarsaLearning():
    def __init__(self, actions,
                 learning_rate=0.01,
                 discount_rate=0.9,
                 decay_rate=0.5,
                 alpha=0.1,
                 epsilon=0.1,
                 debug=True,):

        self.actions = actions
        self.discount_r = discount_rate
        self.decay_r = decay_rate
        self.step_size = alpha
        self.epsilon = epsilon
        self.debug = debug

        self.display_name = "Eligibility Trace Sarsa Learning"
        self.q = {}  # Q table
        self.e = {}  # Eligibility Trace

    def lazy_table_init(self, observation):
        self.q.setdefault(observation, {})
        self.e.setdefault(observation, {})
        for action in self.actions:
            self.e[observation].setdefault(action, 0)
            self.q[observation].setdefault(action, 0)

    def choose_action(self, observation):
        self.lazy_table_init(observation)

        if np.random.uniform() <= (1-self.epsilon):  # Greedy
            return (max(self.q[observation].items(), key=lambda x: x[1]))[0]
        else:  # Epsilon Greedy
            return np.random.choice(np.array(self.actions))

    def learn(self, s, a, r, s_):
        a_ = self.choose_action(s_)
        td_error = r + self.discount_r*self.q[s_][a_] - self.q[s][a]
        self.e[s][a] += 1
        # TODO: Can do this easily via np arrays
        for state in self.q.keys():
            for action in self.q[state]:
                self.q[state][action] += self.step_size * \
                    td_error*self.e[state][action]
                self.e[state][action] = self.discount_r * \
                    self.decay_r*self.e[state][action]
        return s_, a_


if __name__ == "__main__":
    agentXY = [0, 0]
    goalXY = [4, 4]

    wall_shape = np.array([[2, 2], [3, 6]])
    pits = np.array([[6, 3], [1, 4]])
    env = maze_env.Maze(agentXY, goalXY, wall_shape, pits)
    rl_algo = EligibilityTraceSarsaLearning(list(range(env.n_actions)))
