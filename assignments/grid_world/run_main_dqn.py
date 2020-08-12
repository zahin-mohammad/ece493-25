from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from stable_baselines.bench import Monitor
import numpy as np
import sys
from custom_gym import CustomGym
import threading

# Tasks 1-3
tasks = [(np.array([[2, 2], [3, 6]]), np.array([[6, 3], [1, 4]])),
         (np.array([[6, 2], [5, 2], [4, 2], [3, 2], [2, 2], [6, 3], [6, 4], [6, 5],
                    [2, 3], [2, 4], [2, 5]]), []), (np.array([[6, 3], [6, 3], [6, 2], [5, 2], [4, 2], [3, 2], [3, 3],
                                                              [3, 4], [3, 5], [3, 6], [4, 6], [5, 6], [5, 7], [7, 3]]),
                                                    np.array([[1, 3], [0, 5], [7, 7], [8, 5]]))]

experiments = {}
agentXY = [0, 0]
goalXY = [4, 4]

episodes = 2000
time_steps = 1000
gamma = 0.99
learning_rate = 0.001
verbose = 1


def run(double_dqn, task_num):
    title = "Double_Deep_Q" if double_dqn else "Deep_Q"
    env = Monitor(CustomGym(agentXY, goalXY, tasks[task_num][0], tasks[task_num][1], title=f"{title}_Task_{task_num + 1}", ),
                  filename=None)
    model = DQN(MlpPolicy, env, verbose=verbose, gamma=gamma, learning_rate=learning_rate, double_q=bool(double_dqn))
    while len(env.get_episode_rewards()) < episodes:
        model.learn(total_timesteps=time_steps)
    env.save_csv()
    env.destroy()


isDouble = [True, False]
threads = []
if len(sys.argv) > 1:
    threads.append(threading.Thread(target=run, args=(int(sys.argv[1]), 1,)))
else:
    for is_double in [True, False]:
        for i, task in enumerate(tasks):
            threads.append(threading.Thread(target=run, args=(is_double, i,)))

for i, thread in enumerate(threads):
    thread.start()
    print(f'Starting thread {i}')

for i, thread in enumerate(threads):
    thread.join()
    print(f'Stopping thread {i}')
