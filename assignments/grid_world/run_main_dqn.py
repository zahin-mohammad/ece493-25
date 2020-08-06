from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from stable_baselines.bench import Monitor
import numpy as np
import os
import sys
from custom_gym import CustomGym
from stable_baselines import results_plotter
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.callbacks import BaseCallback
import threading

tasks = []
# Task 1
tasks.append((np.array([[2, 2], [3, 6]]), np.array([[6, 3], [1, 4]])))

# Task 2
tasks.append((np.array([[6, 2], [5, 2], [4, 2], [3, 2], [2, 2], [6, 3], [6, 4], [6, 5],
                        [2, 3], [2, 4], [2, 5]]), []))
# Task 3
tasks.append((np.array([[6, 3], [6, 3], [6, 2], [5, 2], [4, 2], [3, 2], [3, 3],
                        [3, 4], [3, 5], [3, 6], [4, 6], [5, 6], [5, 7], [7, 3]]), np.array([[1, 3], [0, 5], [7, 7], [8, 5]])))

experiments = {}
agentXY = [0, 0]
goalXY = [4, 4]

episodes = 2000
time_steps = 1000
gamma = 0.99
learning_rate = 0.001
verbose = 1

def run(isDouble, task):
    title = "Double_Deep_Q" if isDouble else "Deep_Q"
    env = Monitor(CustomGym(agentXY, goalXY,tasks[task][0], tasks[task][1], title=f"{title}_Task_{task+1}",), filename=None)
    model = DQN(MlpPolicy, env, verbose=verbose, gamma=gamma, learning_rate=learning_rate, double_q = bool(isDouble))
    while len(env.get_episode_rewards()) < episodes:
        model.learn(total_timesteps=time_steps)
    env.save_csv()

isDouble = [True, False]
threads = []
if len(sys.argv) > 1:
    for task_num, task in enumerate(tasks):
        threads.append(threading.Thread(target=run, args=(int(sys.argv[1]), task_num,)))
else:
    for isDouble in [True, False]:
        for task_num, task in enumerate(tasks):
            threads.append(threading.Thread(target=run, args=(isDouble, task_num,)))

for i, thread in enumerate(threads):
    thread.start()
    print(f'Starting thread {i}')

for i, thread in enumerate(threads):
    thread.join()
    print(f'Stopping thread {i}')