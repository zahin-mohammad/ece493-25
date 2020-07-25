import numpy as np
import maze_env


class DynamicProgrammingAlgorithm:
    @staticmethod
    def get_tkInter_coords(center_cords):
        return (center_cords[0]-15, center_cords[1]-15, center_cords[0]+15, center_cords[1]+15)

    def __init__(self, actions, environment):
        self.actions = actions
        self.action_map = {
            0: (0, -40, 0, -40), 1: (0, 40, 0, 40), 2: (40, 0, 40, 0), 3: (-40, 0, -40, 0)}
        self.env = {}
        self.walls = {}

        self.init_env(environment)
        self.policy = {}
        self.values = {}

    # First value is the actual next state.
    # Second value used for environment reward function.
    def next_state(self, o_state, action):
        n_state = []
        for i, addition in enumerate(self.action_map[action]):
            n_state.append(o_state[i]+addition)
        n_state = tuple(n_state)

        if n_state in self.walls:
            return o_state, n_state

        for point in n_state:
            if 0 > point or point > maze_env.MAZE_H*maze_env.UNIT:
                return o_state, o_state
        return n_state, n_state

    def init_env(self, maze_environment):
        self.walls = set([tuple(maze_environment.canvas.coords(w))
                          for w in maze_environment.wallblocks])

        for x in range(0, maze_env.MAZE_W):
            for y in range(0, maze_env.MAZE_H):
                center = maze_env.origin + \
                    np.array([maze_env.UNIT * x, maze_env.UNIT*y])
                # walls are not states agent can be in
                coordinates = DynamicProgrammingAlgorithm.get_tkInter_coords(
                    (center[0], center[1]))
                if coordinates in self.walls:
                    continue
                self.env.setdefault(coordinates, {})

                for action in self.actions:
                    next_state, possible_wall_state = self.next_state(
                        coordinates, action)
                    reward, done, __ = maze_environment.computeReward(
                        list(coordinates), action, list(possible_wall_state))
                    self.env[coordinates][action] = (next_state, reward, done)
