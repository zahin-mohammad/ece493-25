import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
import time
import threading
from maze_env import Maze

DEBUG = 1


def debug(debuglevel, msg, **kwargs):
    if debuglevel <= DEBUG:
        if 'printNow' in kwargs:
            if kwargs['printNow']:
                print(msg)
        else:
            print(msg)


def update(task_num, env, RL, data,
           sim_speed=0.001,
           showRender=False,
           renderEveryNth=1000,
           printEveryNth=500,
           episodes=100):
    global_reward = np.zeros(episodes)
    data['global_reward'] = global_reward
    data[RL.display_name] = []

    for episode in range(episodes):
        start = time.time()
        t = 0

        # initial state
        state = env.reset(value=0) if episode == 0 else env.reset()
        # RL choose action based on state
        action = RL.choose_action(str(state))

        while True:
            # fresh env
            if(showRender or (episode % renderEveryNth) == 0):
                env.render(sim_speed)

            # RL take action and get next state and reward
            state_, reward, done = env.step(action)
            global_reward[episode] += reward
            debug(2, f'state(ep:{episode},t:{t})={state}')
            debug(
                2, f'reward_{reward} = total return_t = {global_reward[episode]} Mean50 = {np.mean(global_reward[-50:])}')

            # RL learn from this transition
            # and determine next state and action
            # print(str(state))
            state, action = RL.learn(str(state), action, reward, str(state_))

            # break while loop when end of this episode
            if done:
                break
            else:
                t = t+1
        end = time.time()
        data[RL.display_name].append(end-start)
        debug(1, "({}) Episode {}: Length={}  Total return = {} ".format(RL.display_name, episode, t,
                                                                         global_reward[episode], global_reward[episode]), printNow=(episode % printEveryNth == 0))
        if(episode >= 100):
            debug(1, "    Median100={} Variance100={}".format(np.median(global_reward[episode-100:episode]), np.var(
                global_reward[episode-100:episode])), printNow=(episode % printEveryNth == 0))

    # end of game
    print(f'game over -- Algorithm {RL.display_name} completed')
    env.destroy()
