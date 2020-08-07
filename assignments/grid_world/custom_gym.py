from gym import Env, spaces
from maze_env import Maze
import random
import numpy as np
import time
import csv


class CustomGym(Env):
    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    The methods are accessed publicly as "step", "reset", etc...
    """

    def __init__(self, agentXY, goalXY, walls=[], pits=[], title = 'Maze',):
        super(CustomGym, self).__init__()
        self.env = Maze(agentXY, goalXY, walls, pits, title)
        self.title = title
        self.action_space = spaces.Discrete(self.env.n_actions)
        self.observation_space = spaces.Box(low=0, high=0, shape=(4,), dtype=np.float32)
        
        self.rewards=[[]]
        self.variance = []
        self.median = []


    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        s_, reward, done = self.env.step(action)

        self.rewards[-1].append(reward)
        if done:
            self.variance.append(np.var(self.rewards[-1]))
            self.median.append(np.median(self.rewards[-1]))
            self.rewards.append([])

        return s_, reward, done, {}

    def render(self, mode='human'):
        self.env.render()

    def reset(self, value=1, resetAgent=True):
        return self.env.reset()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        np.random.seed(10)
        random.seed(10)
        return
    
    def save_csv(self):
        with open(f"./data/{self.title}_rewards_{time.time()}","w+") as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(self.rewards)
        with open(f"./data/{self.title}_variance_{time.time()}","w+") as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            for var in self.variance:
                csvWriter.writerow([var])
        with open(f"./data/{self.title}_median_{time.time()}","w+") as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            for med in self.median:
                csvWriter.writerow([med])
    
    def destroy(self):
        self.env.destroy()

   









 







