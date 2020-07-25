import time
import sys
import matplotlib.pyplot as plt
import numpy as np

from run_main import update
from maze_env import Maze
from real_value_ga import RealValueGas

from RL_brain_expected_sarsa import ExpectedSarsaLearning as rlalg1
from RL_brain_double_q_learning import DoubleQLearning as rlalg2
from RL_brain_eligibility_trace_sarsa import EligibilityTraceSarsaLearning as rlalg3


def simulation(
    generation_count,
    init_population,
    parent_selection,
    crossover,
    mutation,
    survivor_selection,
    best_of_generation,
    debug=False
):
    population = init_population()
    best_per_generation = []
    for i in range(generation_count):
        start = time.time()
        parents = parent_selection(population)
        children = mutation(crossover(parents))
        population = survivor_selection(children, population)
        best_fitness, best_individual = best_of_generation(population)
        best_per_generation.append(best_fitness)
        end = time.time()
        if debug:
            print(
                f'For gen: {i+1} fitness:{best_per_generation[-1]} took: {end-start} s', flush=True)
            print(f'\tbest_individual:{best_individual}', flush=True)
            print(f'\tpop-size: {len(population)}')

    return best_per_generation


if __name__ == "__main__":
    showRender = True
    episodes = 10
    renderEveryNth = 1000
    printEveryNth = 500
    do_plot_rewards = True

    # Env 3
    agentXY = [0, 0]
    goalXY = [4, 4]
    tasks = []
    tasks.append((np.array([[6, 3], [6, 3], [6, 2], [5, 2], [4, 2], [3, 2], [3, 3],
                            [3, 4], [3, 5], [3, 6], [4, 6], [5, 6], [5, 7], [7, 3]]), np.array([[1, 3], [0, 5], [7, 7], [8, 5]])))
    env = Maze(agentXY, goalXY, tasks[0][0], tasks[0][1])

    def rl_brain(rl):

        data = {}
        env.after(10, update(tasks[0], env, rl, data, episodes))
        env.mainloop()
        max_rewards = np.max(
            data['global_reward'])
        median_rewards = np.median(
            data['global_reward'][-100:])
        print("{} : max reward = {} medLast100={}".format(RL.display_name,
                                                          max_rewards,
                                                          median_rewards))
        return np.array([max_rewards, median_rewards])

    double_q_gene_range = [
        (0.01, 0.99),   # learning_rate
        (0.01, 0.99),   # discount_rate
        (0.01, 0.99)    # epsilon greedy
    ]

    def double_q_fitness(learning_rate, discount_rate, epsilon):
        rl = rlalg2(
            list(range(env.n_actions)),
            learning_rate=learning_rate,
            discount_rate=discount_rate,
            epsilon=epsilon)
        rewards = rl_brain(rl)
        sum_rewards = np.sum(-1*rewards)
        ret_val = 1.0/sum_rewards if sum_rewards != 0 else 0
        print(ret_val)
        return ret_val

    expected_sarsa_gene_range = [
        (0.01, 0.99),   # learning_rate
        (0.01, 0.99),   # discount_rate
        (0.01, 0.99)    # epsilon greedy
    ]

    def expected_sarsa_fitness():
        rl = rlalg1(list(range(env.n_actions)))
        rewards = rl_brain(rl)
        return 1.0/np.sum(-1*rewards)

    eligibility_sarsa_gene_range = [
        (0.01, 0.99),   # learning_rate
        (0.01, 0.99),   # discount_rate
        (0.01, 0.99),   # decay_rate
        (0.01, 0.99),   # alpha (step size)
        (0.01, 0.99),   # epsilon greedy
    ]

    def eligibility_sarsa_fitness():
        rl = rlalg3(list(range(env.n_actions)))
        rewards = rl_brain(rl)
        return 1.0/np.sum(-1*rewards)

    real_value_gas = RealValueGas(double_q_gene_range, double_q_fitness)
    simulation(
        generation_count=real_value_gas.generation_count,
        init_population=real_value_gas.init_population,
        parent_selection=real_value_gas.parent_selection,
        crossover=real_value_gas.crossover,
        mutation=real_value_gas.mutation,
        survivor_selection=real_value_gas.survivor_selection,
        best_of_generation=real_value_gas.best_of_population,
        debug=True

    )
