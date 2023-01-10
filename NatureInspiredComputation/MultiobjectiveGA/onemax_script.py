import random
import numpy as np

# param: gen_num means the number of solution
def generate(gen_num):
    population_list = []
    for i in range(gen_num):
        individual = []
        for j in range(15):
            individual.append(random.randint(0, 1))

        population_list.append(individual)

    return population_list

def fitness(individual):
    return sum(individual)

# tour_size can not larger than population_length
def tournament(tour_size, population_list):
    tour_list :list = []
    for i in range(tour_size):
        rd_num = random.randint(0, len(population_list)-1)
        tour_list.append(population_list[rd_num])

    parent = max(tour_list, key=lambda individual: fitness(individual)).copy()

    return parent

def crossover(i1,i2):
    check_point = random.randint(0,len(i1)-1)
    new_i1 = i1[0:check_point] + i2[check_point:]
    new_i2 = i2[0:check_point] + i1[check_point:]
    return [new_i1, new_i2]

def mutate(individual,size=1,pro=1):
    new_individual = individual.copy()
    for i in range(size):
        rd_pro = np.random.rand()
        if (rd_pro < pro):
            rd_mut_point = random.randint(0, len(new_individual) - 1)
            new_individual[rd_mut_point] = 1 - new_individual[rd_mut_point]
        else:
            continue
    return new_individual

def replacement(population_list,individual):
    population_list.sort(key=lambda individual:fitness(individual))
    # replace the worst individual
    if fitness(individual) > fitness(population_list[0]):
        population_list[0] = individual

# to find out the max fitness in the population
def get_population_max_fitness(population_list):
    return max(fitness(individual) for individual in population_list)

def run_ga(population_size,tournament_size,mutation_rate,mutation_size,crossover_rate,it):
    
    iterations = []

    # simulate 100 time ,calculate the probability of success
    for i in range(it):
        record_iter = 0

        # generation
        pop_list = generate(population_size)

        while True:

            #print("picking parents")
            # tournament -- pick parents
            parent_list = []
            for i in range(tournament_size):
                parent_list.append(tournament(3, pop_list))

            #print("crossing parents")
            # crossover
            child_list = []
            rdn = np.random.rand()
            if rdn < crossover_rate:
                for i in range(0, len(parent_list), 2):
                    child_list.extend(crossover(parent_list[i], parent_list[i + 1]))

            #print("mutating")
            # mutation
            mu_child_list = []
            for child in child_list:
                mu_child_list.append(mutate(child, size=mutation_size,pro=mutation_rate))

            #print("replacing")
            # replacement
            for mu_child in mu_child_list:
                replacement(pop_list, mu_child)

            # get_population_max_fitness
            max_population_fitness = get_population_max_fitness(pop_list)

            #print(max_population_fitness)
            if max_population_fitness == 15 or record_iter > 1500:
                break
            else:
                record_iter += 1

        iterations.append(record_iter)

    return iterations

if __name__ == '__main__':

    [15, 6, 0.3896936257590489, 0, 0.5555777866489992]
    [5, 2, 0.99, 9, 0.4031313884250146]

    population_size = 5
    tournament_size = 2
    mutation_rate = 0.9699201486715177
    mutation_size = 9
    crossover_rate = 0.4031313884250146

    print(run_ga(population_size,tournament_size,mutation_rate,mutation_size,crossover_rate))

