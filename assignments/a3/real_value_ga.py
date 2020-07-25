import numpy as np
import heapq
import random


class RealValueGas():
    def __init__(self,
                 gene_range,
                 fitness_function,
                 population_size=50,
                 generation_count=150,
                 prob_crossover=0.6,
                 prob_mutation=0.25,
                 survival_count=2,
                 init_pop=[]):
        super().__init__()
        self.gene_range = gene_range
        self.fitness_fn = fitness_function
        self.population_size = population_size
        self.generation_count = generation_count
        self.p_c = prob_crossover
        self.p_m = prob_mutation
        self.survival_count = survival_count
        self.cache = {}
        self.init_pop = init_pop

    # Meant to keep same initial population if possible to test different parameters
    def reset(self, population_size=50,
              generation_count=150,
              prob_crossover=0.6,
              prob_mutation=0.25,
              survival_count=2,
              init_pop=None):
        if population_size != self.population_size:
            init_pop = []
        self.population_size = population_size
        self.generation_count = generation_count
        self.p_c = prob_crossover
        self.p_m = prob_mutation
        self.survival_count = survival_count
        self.cache = {}

    def create_individual_genes(self):
        return np.array([np.random.uniform(low=low, high=high) for low, high in self.gene_range])

    def init_population(self):
        if len(self.init_pop) == 0:
            self.init_pop = [tuple(np.round(self.create_individual_genes(), 2))
                             for i in range(self.population_size)]
        return self.init_pop

    def fitness(self, individual):
        # Do not use setdefault, it has performance issues
        if individual in self.cache:
            return self.cache[individual]
        fitness = 0.0

        # try:
        fitness = self.fitness_fn(*individual)
        # except:
        # pass

        self.cache[individual] = fitness
        return fitness

    def mutation(self, individuals):
        for i, _ in enumerate(individuals):
            genes = list(individuals[i])
            # Uniform Mutation
            for k, gene in enumerate(individuals[i]):
                if np.random.uniform() <= (self.p_m):
                    genes[k] = np.random.uniform(
                        low=self.gene_range[k][0], high=self.gene_range[k][1])
            individuals[i] = tuple(np.round(np.array(genes), 2))
        return individuals

    def crossover(self, parents):
        children = []
        random.shuffle(parents)

        # Uniform Crossover
        for i in range(1, len(parents), 2):
            child_a, child_b = list(parents[i-1]), list(parents[i])
            for k, gene in enumerate(child_a):
                if np.random.uniform() <= (self.p_c):
                    child_a[k], child_b[k] = child_b[k], child_a[k]
            child_a, child_b = tuple(child_a), tuple(child_b)
            children.extend([child_a, child_b])
        return children

    def survivor_selection(self, next_generation, current_generation):
        # keep a pop of population_size always
        # remove survivor count number of worst individuals from new pop and
        # replace with survivor count of best individuaLS old pop
        return heapq.nsmallest(
            self.population_size - self.survival_count, next_generation, key=self.fitness) + heapq.nlargest(
            self.survival_count, current_generation, key=self.fitness)

    def parent_selection(self, individuals):
        fitnesses = np.array([self.fitness(individual)
                              for individual in individuals])
        selection_prob = fitnesses / fitnesses.sum()
        # Spin roulette wheel until we get population size number of individuals
        new_gen = [individuals[i] for i in np.random.choice(
            len(individuals), len(individuals), p=selection_prob)]
        return new_gen

    def best_of_population(self, population):
        best_individual = max(population, key=self.fitness)
        return self.fitness(best_individual), best_individual
