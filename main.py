import numpy as np 
import random
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import time

class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return "({0}, {1})".format(self.x, self.y)
    
    def distance(self, city):
        dist_x = abs(self.x - city.x)
        dist_y = abs(self.y - city.y)
        return np.sqrt(dist_x**2 + dist_y**2)


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def route_distance(self):
        if self.distance == 0:
            path_dist = 0
            for i in range(0, len(self.route)):
                origin = self.route[i]
                dest = self.route[i + 1] if len(self.route) > i + 1 else self.route[0]
                path_dist += origin.distance(dest)
            self.distance = path_dist
        return self.distance
    
    def route_fitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.route_distance())
        return self.fitness


def initial_population(pop_size, cities):
    return [random.sample(cities, len(cities)) for i in range(0, pop_size)]


def sort_routes(population):
    fitnesses = {i: Fitness(individual).route_fitness() for (i, individual) in enumerate(population)}
    return sorted(fitnesses.items(), key= lambda x:x[1], reverse=True)


def selection(sorted_pop, elite_size):
    selected = []
    fitnesses = [y for (x, y) in sorted_pop]
    cumulative_freq = np.cumsum(fitnesses)/np.sum(fitnesses)
    for i in range(0, elite_size):
        selected.append(sorted_pop[i][0])
    for i in range(0, len(sorted_pop) - elite_size):
        pick = random.random()
        for i in range(0, len(sorted_pop)):
            if pick <= cumulative_freq[i]:
                selected.append(sorted_pop[i][0])
                break
    return selected


def mating_pool(population, selected):
    pool = []
    for i in range(0, len(selected)):
        pool.append(population[selected[i]])
    return pool


def breed(parent1, parent2):
    child_p1 = []
    child_p2 = []

    gene_a = int(random.random()*len(parent1))
    gene_b = int(random.random()*len(parent1))

    start = min(gene_a, gene_b)
    end = max(gene_a, gene_b)

    for i in range(start, end):
        child_p1.append(parent1[i])
    
    child_p2 = [item for item in parent2 if item not in child_p1]

    return child_p1 + child_p2


def breed_population(matingpool, elite_size):
    children = []
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, elite_size):
        children.append(matingpool[i])
    
    for i in range(0, len(matingpool) - elite_size):
        child = breed(pool[i], pool[len(matingpool)- i - 1])
        children.append(child)
    return children


def mutate(individual, mutation_rate):
    for swapped in range(len(individual)):
        if (random.random() < mutation_rate):
            to_swap = int(random.random() * len(individual))
            city1 = individual[swapped]
            city2 = individual[to_swap]
            individual[swapped] = city2
            individual[to_swap] = city1
    return individual


def mutate_population(population, mutation_rate):
    mutated = [mutate(population[i], mutation_rate) for i in range(0, len(population))]
    return mutated


def next_generation(current_gen, elite_size, mutation_rate):
    sorted_pop = sort_routes(current_gen)
    selected = selection(sorted_pop, elite_size)
    matingpool = mating_pool(current_gen, selected)
    children = breed_population(matingpool, elite_size)
    next_gen = mutate_population(children, mutation_rate)
    return next_gen


def genetic_algorithm(population, population_size, elite_size, mutation_rate, generations):
    pop = initial_population(population_size, population)
    for i in range(0, generations):
        pop = next_generation(pop, elite_size, mutation_rate)
    return pop[sort_routes(pop)[0][0]]

def genetic_algorithm_progress(population, population_size, elite_size, mutation_rate, generations):
    pop = initial_population(population_size, population)
    progress = []
    progress.append(1 / sort_routes(pop)[0][1])
    for i in range(0, generations):
        pop = next_generation(pop, elite_size, mutation_rate)
        progress.append(1 / sort_routes(pop)[0][1])
    return pop[sort_routes(pop)[0][0]], progress


def show_solution(sol, progress):
    if progress:
        plt.plot(progress)
        plt.ylabel('Distance')
        plt.xlabel('Generation')
        plt.show()
    print(sol)
    cities = [(c.x, c.y) for c in sol]
    cities.append(cities[0])
    x, y = zip(*cities)
    plt.plot(x, y, marker='o', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=4)
    plt.show()  


def tsp(n):
    start_time = time.clock()
    cities = [City(int(random.random() * 200), int(random.random() * 200)) for i in range(0, n)]
    sol, progress = genetic_algorithm_progress(cities, 50, 10, 0.005, 250)
    print("GA: " + format(time.clock() - start_time, '.5f') + " seconds")
    show_solution(sol, progress)


if __name__ == '__main__':
    tsp(10)