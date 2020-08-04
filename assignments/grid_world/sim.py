import numpy as np
import threading
import sys

from run_main import update, debug
from maze_env import Maze
from plots import *
# A3 MC
from monte_carlo.MC_policy_gradient import PolicyGradientLearning
# A3 TD
from learning_algorithms.RL_brain_double_q_learning import DoubleQLearning
from learning_algorithms.RL_brain_eligibility_trace_sarsa import EligibilityTraceSarsaLearning
from learning_algorithms.RL_brain_expected_sarsa import ExpectedSarsaLearning
# A2 TD
from learning_algorithms.RL_brain_q_learning import QLearning
from learning_algorithms.RL_brain_sarsa import SarsaLearning
# A2 dp
from dynamic_programming.DP_brain_PI import PolicyIteration
from dynamic_programming.DP_brain_VI import ValueIteration


# showRender = False
episodes = 5000
# renderEveryNth = 10000
# printEveryNth = 10000
do_plot_rewards = True

# Task Specifications

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


def append_data(task_num, env, rl, data):
    sem.acquire()
    experiments.setdefault(task_num, [])
    experiments[task_num].append((env, rl, data))
    sem.release()


def do_algorithm(rl, task, env):
    data = {}
    env.after(10, update(env, rl, data,
                         showRender=True,
                         episodes=episodes,
                         renderEveryNth=100,
                         printEveryNth=100,
                         ))
    env.mainloop()
    append_data(task, env, rl, data)


def value_iteration(task):
    env = Maze(agentXY, goalXY, tasks[task][0], tasks[task][1], title=f"Value Iteration task {task+1}")
    rl = ValueIteration(list(range(env.n_actions)), env)
    do_algorithm(rl, task, env)


def policy_iteration(task):
    env = Maze(agentXY, goalXY, tasks[task][0], tasks[task][1], title=f"Policy Iteration task {task+1}")
    rl = PolicyIteration(list(range(env.n_actions)), env)
    do_algorithm(rl, task, env)


def sarsa_learning(task):
    env = Maze(agentXY, goalXY, tasks[task][0], tasks[task][1], title=f"Sarsa Learning task {task+1}")
    rl = SarsaLearning(list(range(env.n_actions)))
    do_algorithm(rl, task, env)


def expected_sarsa_learning(task):
    env = Maze(agentXY, goalXY, tasks[task][0], tasks[task][1], title=f"Expected Sarsa Learning task {task+1}")
    rl = ExpectedSarsaLearning(list(range(env.n_actions)))
    do_algorithm(rl, task, env)


def q_learning(task):
    env = Maze(agentXY, goalXY, tasks[task][0], tasks[task][1], title=f"Q Learning task {task+1}")
    rl = QLearning(list(range(env.n_actions)))
    do_algorithm(rl, task, env)


def double_q_learning(task):
    env = Maze(agentXY, goalXY, tasks[task][0], tasks[task][1], title=f"Double Q Learning task {task+1}")
    rl = DoubleQLearning(list(range(env.n_actions)))
    do_algorithm(rl, task, env)

def eligibility_trace_learning(task):
    env = Maze(agentXY, goalXY, tasks[task][0], tasks[task][1], title=f"Eligibility Trace Learning task {task+1}")
    rl = EligibilityTraceSarsaLearning(list(range(env.n_actions)))
    do_algorithm(rl, task, env)


def policy_gradient_learning(task):
    env = Maze(agentXY, goalXY, tasks[task][0], tasks[task][1], title=f"Policy Gradient Learning task {task+1}")
    rl = PolicyGradientLearning(list(range(env.n_actions)))
    do_algorithm(rl, task, env)


# algos = [expected_sarsa_learning]

algos = [sarsa_learning, q_learning, expected_sarsa_learning, eligibility_trace_learning, double_q_learning, policy_gradient_learning]



threads, sem = [], threading.Semaphore()
if len(sys.argv) > 1:
    algo = algos[int(sys.argv[1])]
    for task_num, task in enumerate(tasks):
        print(f"running {algo.__name__} {task_num}")
        threads.append(threading.Thread(target=algo, args=(task_num,)))
else:   
    for algo in algos:
        for task_num, task in enumerate(tasks):
            print(f"running {algo.__name__} {task_num}")
            threads.append(threading.Thread(target=algo, args=(task_num,)))

for i, thread in enumerate(threads):
    debug(1, f'Starting thread {i}')
    thread.start()
for i, thread in enumerate(threads):
    debug(1, f'Stopping thread {i}')
    thread.join()

print("All experiments complete")
for task_num, experiment_list in experiments.items():
    for env, RL, data in experiment_list:
        print("Task {}, {} : max reward = {} medLast100={} varLast100={}".format(task_num, RL.display_name, np.max(
            data['global_reward']), np.median(data['global_reward'][-100:]), np.var(data['global_reward'][-100:])))

if(do_plot_rewards):
    # Simple plot of return for each episode and algorithm, you can make more informative plots
    if len(sys.argv) > 1:
        plot_rewards(experiments, algos[int(sys.argv[1])].__name__)
