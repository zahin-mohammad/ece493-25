import matplotlib.pyplot as plt
import numpy as np

PLOT_DIR = "./plots/"


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
    fig.savefig(f"{PLOT_DIR}reward_graph_by_task.png", dpi=500)
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
    fig.savefig(f"{PLOT_DIR}reward_graph_by_algorithm.png", dpi=500)


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
    fig.savefig(f"{PLOT_DIR}time_graph.png", dpi=500)
    # plt.show()
