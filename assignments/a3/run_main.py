import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
import time
import threading

from maze_env import Maze
from RL_brain_expected_sarsa import ExpectedSarsaLearning as rlalg1
from RL_brain_double_q_learning import DoubleQLearning as rlalg2
from RL_brain_eligibility_trace_sarsa import EligibilityTraceSarsaLearning as rlalg3


DEBUG = 1


def debug(debuglevel, msg, **kwargs):
    if debuglevel <= DEBUG:
        if 'printNow' in kwargs:
            if kwargs['printNow']:
                print(msg)
        else:
            print(msg)


def plot_rewards(task_experiments):
    fig, axs = plt.subplots(2, 2)
    color_list = ['blue', 'green', 'red', 'black', 'magenta']
    label_list = []
    axs_keys = {0: (0, 0), 1: (0, 1), 2: (1, 0)}
    for task, experiments in task_experiments.items():
        for i, (env, RL, data) in enumerate(experiments):
            x_values = range(len(data['global_reward']))
            label_list.append(RL.display_name)
            y_values = data['global_reward']

            axs[axs_keys[task][0], axs_keys[task][1]].plot(
                x_values, y_values, c=color_list[i], label=label_list[-1])
        axs[axs_keys[task][0], axs_keys[task][1]].legend(label_list)
        axs[axs_keys[task][0], axs_keys[task][1]].set_title(f'Task {task+1}')

    fig.text(0.5, 0.04, 'Episodes', ha='center', fontsize=18)
    fig.text(0.04, 0.5, 'Rewards', va='center',
             rotation='vertical', fontsize=18)
    fig.suptitle('DP and Learning Algorithm Episodes vs Rewards', fontsize=18)

    fig.set_size_inches((8.5, 11), forward=False)
    fig.savefig("reward_graph_by_task.png", dpi=500)
    # plt.show()


def plot_individual_plots(task_experiments):
    fig, axs = plt.subplots(2, 2)
    color_list = ['blue', 'green', 'red', 'black', 'magenta']
    axs_keys = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
    name_to_tasks = {}

    for task, experiments in task_experiments.items():
        for i, (env, RL, data) in enumerate(experiments):
            name_to_tasks.setdefault(RL.display_name, {})
            name_to_tasks[RL.display_name][task] = (env, RL, data)

    label_list = []
    for j, name in enumerate(name_to_tasks.keys()):
        for i, item in enumerate(name_to_tasks[name].items()):
            task, experiment = item
            env, RL, data = experiment
            x_values = range(len(data['global_reward']))
            label_list.append(task)
            y_values = data['global_reward']
            axs[axs_keys[j][0], axs_keys[j][1]].plot(
                x_values, y_values, c=color_list[i], label=label_list[-1])
        axs[axs_keys[j][0], axs_keys[j][1]].legend(label_list)
        axs[axs_keys[j][0], axs_keys[j][1]].set_title(f'{RL.display_name}')

    fig.text(0.5, 0.04, 'Episodes', ha='center', fontsize=18)
    fig.text(0.04, 0.5, 'Rewards', va='center',
             rotation='vertical', fontsize=18)
    fig.suptitle('DP and Learning Algorithm Episodes vs Rewards', fontsize=18)

    fig.set_size_inches((8.5, 11), forward=False)
    fig.savefig("reward_graph_by_algorithm.png", dpi=500)


def plot_time(task_experiments):
    fig, ax = plt.subplots()
    color_list = ['blue', 'green', 'red', 'black', 'magenta']
    label_list = []
    axs_keys = {0: (0, 0), 1: (0, 1), 2: (1, 0)}

    name_to_time = {}
    for task, experiments in task_experiments.items():
        for i, (env, RL, data) in enumerate(experiments):
            name_to_time.setdefault(RL.display_name, [])
            name_to_time[RL.display_name].extend(data[RL.display_name])

    labels = []
    y_values = []
    for name, times in name_to_time.items():
        labels.append(name)
        y_values.append((sum(times)/len(times))*1000)
    x = np.arange(len(labels))

    width = 0.35
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    rects1 = ax.bar(x, y_values, width)

    fig.text(0.5, 0.04, 'Algorithms', ha='center', fontsize=18)
    fig.text(0.04, 0.5, 'Time [ms]', va='center',
             rotation='vertical', fontsize=18)
    fig.suptitle('Average time/episode For All Tasks', fontsize=18)

    fig.set_size_inches((8.5, 11), forward=False)
    fig.savefig("time_graph.png", dpi=500)
    # plt.show()


def update(task_num, env, RL, data,
           sim_speed=0.01,
           showRender=True,
           renderEveryNth=1000,
           printEveryNth=500,
           episodes=50):
    global_reward = np.zeros(episodes)
    data['global_reward'] = global_reward
    data[RL.display_name] = []

    for episode in range(episodes):
        start = time.time()
        t = 0
        # initial state
        if episode == 0:
            state = env.reset(value=0)
        else:
            state = env.reset()

        # RL choose action based on state
        action = RL.choose_action(str(state))
        while True:
            # fresh env
            if(showRender or (episode % renderEveryNth) == 0):
                env.render(sim_speed)

            # RL take action and get next state and reward
            state_, reward, done = env.step(action)
            global_reward[episode] += reward
            debug(2, 'state(ep:{},t:{})={}'.format(episode, t, state))
            debug(2, 'reward_{}=  total return_t ={} Mean50={}'.format(
                reward, global_reward[episode], np.mean(global_reward[-50:])))

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
    print('game over -- Algorithm {} completed'.format(RL.display_name))
    env.destroy()


if __name__ == "__main__":
    sim_speed = 0.05

    # Example Short Fast for Debugging
    showRender = False
    episodes = 10
    renderEveryNth = 1000
    printEveryNth = 500
    do_plot_rewards = True

    # Example Full Run, you may need to run longer
    # showRender = True
    # episodes = 2000
    # renderEveryNth = 1000
    # printEveryNth = 500
    # do_plot_rewards = True

    # if(len(sys.argv) > 1):
    #     episodes = int(sys.argv[1])
    # if(len(sys.argv) > 2):
    #     showRender = sys.argv[2] in ['true', 'True', 'T', 't']
    # if(len(sys.argv) > 3):
    #     datafile = sys.argv[3]

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

    def expected_sarsa_learning(task):
        env = Maze(agentXY, goalXY, tasks[task][0], tasks[task][0])
        rl = rlalg1(list(range(env.n_actions)))
        data = {}
        env.after(10, update(task, env, rl, data, episodes))
        env.mainloop()
        append_data(task, env, rl, data)

    def double_q_learning(task):
        env = Maze(agentXY, goalXY, tasks[task][0], tasks[task][0])
        rl = rlalg2(list(range(env.n_actions)))
        data = {}
        env.after(10, update(task, env, rl, data, episodes))
        env.mainloop()
        append_data(task, env, rl, data)

    def eligibility_trace_sarsa_learning(task):
        env = Maze(agentXY, goalXY, tasks[task][0], tasks[task][0])
        rl = rlalg3(list(range(env.n_actions)))
        data = {}
        env.after(10, update(task, env, rl, data, episodes))
        env.mainloop()
        append_data(task, env, rl, data)

    def run(method, task_num):
        if method == 0:
            expected_sarsa_learning(task_num)
        elif method == 1:
            double_q_learning(task_num)
        elif method == 2:
            eligibility_trace_sarsa_learning(task_num)

    # algo = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != '' else 0
    # for task_num, task in enumerate(tasks):
    #     run(algo, task_num)

    threads = []
    sem = threading.Semaphore()
    algos = [expected_sarsa_learning, double_q_learning,
             eligibility_trace_sarsa_learning]
    for i in range(len(algos)):
        for task_num, task in enumerate(tasks):
            threads.append(threading.Thread(target=run, args=(i, task_num)))

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
        plot_rewards(experiments)
        plot_time(experiments)
        plot_individual_plots(experiments)
