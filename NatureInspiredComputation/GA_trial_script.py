import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from IPython.display import display
from tqdm import tqdm
import sys

"""
This is getting the bags data for our algorithm and storing it in 'bags'; a dictionary which we can reference later.
"""

van_capacity = 285
bags = {}
with open('BankProblem.txt','r') as f:
    lines = f.readlines()
    for i in range(1,len(lines),3):
        bags[lines[i].strip()[:-1]] = {'weight': float(lines[i+1][10:]), 'value': float(lines[i+2][9:])}

keys = list(bags.keys())

"""
Defining the functions for our GA including our GA operators.
"""

def list_2_string(list):
    strin = ''
    for i in list:
        strin = strin + str(i)
    return strin

def sln_value_weight(chromosome):

    """returns value and weight of a candidate solution"""

    value = 0
    weight = 0

    for i in range(len(chromosome)):
        if int(chromosome[i]) == 1:
            value += bags[keys[i]]['value']
            weight += bags[keys[i]]['weight']

    return value , weight


def scatter_solution(chromosome):

    """Takes in a bit string argument and presents which points are active or not on a scatter plot."""

    off_on = ['red','green']
    dict_keys = list(bags.keys())

    for i in range(len(chromosome)):
        x = float(bags[dict_keys[i]]['weight'])
        y = float(bags[dict_keys[i]]['value'])
        plt.scatter(x,y,color=off_on[int(chromosome[i])],alpha=0.7)
    plt.xlabel('weight (Kg)')
    plt.ylabel('value (£)')
    plt.show()

def fitness_func(chromosome):

    """outputs the fitness of the candidate solution."""

    value , weight = sln_value_weight(chromosome)

    if weight > van_capacity:
        excess = weight - van_capacity
        return  -excess
    else:
        return value

def generate_population(n,l=100):
    
    """Generates a population of n chromosomes with length l"""
    
    #This actually generates the list of solutions.
    solutions_list = np.array([ np.random.randint(0,2,l) for i in range(n) ])
    candidate_solutions = np.array([])

    #This turns each solution into a string.
    for i in range(len(solutions_list)):
        x = list_2_string(solutions_list[i])
        candidate_solutions = np.append(candidate_solutions,x)
    
    return candidate_solutions

def rank_population(population):

    """
    Returns a dataframe of the ranked chromosomes with the bit string, rank, and fitness for use in the genetic algorithm
    """

    #Making the population a dataframe
    ranked_population = pd.DataFrame(columns=['chromosome','fitness','rank'])

    for i in range(len(population)):

        chrom_row = pd.DataFrame({'chromosome':population[i],'fitness':fitness_func(population[i]),'rank':''},index=[0])
        ranked_population = pd.concat([ranked_population,chrom_row],ignore_index=True)
        
    #Sort the population based on fitness
    ranked_population = ranked_population.sort_values(by=['fitness'],ascending=False)
    ranked_population.index = np.arange(0,len(ranked_population))

    #Initialise the rank as the highest rank in the population i.e. the length
    rank = len(population)

    for i in range(len(ranked_population)):

        if ranked_population['rank'][i] != '' :
            continue
        else:
            
            index_to_rank = np.where(ranked_population['fitness'] == ranked_population['fitness'][i])
            ranked_population.loc[:,'rank'].iloc[index_to_rank] = rank
            rank -= len(index_to_rank)

    return ranked_population
    
def crossover(parent_1,parent_2):

    """Crosses over two parent strings at random location and returns two child bit strings"""

    cop = np.random.randint(0,len(parent_1))
    child_1 = parent_1[:cop] + parent_2[cop:]
    child_2 = parent_2[:cop] + parent_1[cop:]

    return child_1 , child_2

def mutate(candidate):

    """Randomyl selects an index and switches its value between 1 and 0"""

    index = np.random.randint(0,100)

    if candidate[index] == str(1):
        return candidate[:index] + str(0) + candidate[index+1:]
    else:
        return candidate[:index] + str(1) + candidate[index+1:]

def replacement(ranked_population, child_1,child_2):

    """Takes ranked population and two children. Returns np array of bit strings as the new population."""
    
    population = np.array(ranked_population['chromosome'])
    child_fitness = [fitness_func(child_1),fitness_func(child_2)]
    
    child_1_index = np.argmax(child_fitness[0] > ranked_population['fitness'])
    child_2_index = np.argmax(child_fitness[1] > ranked_population['fitness'])


    population = np.insert(population,child_1_index,child_1)
    population = np.insert(population,child_2_index,child_2)


    population = population[:-2]

    return population

def tournament_selection(ranked_population,tournament_size=2):

    """
    Function to pick two parents from two tournaments with specified size
    We expect the ranked_population to be a dataframe from the rank_population function
    """

    parents=[]

    probability = np.array([ ranked_population['rank'].iloc[i]/ranked_population['rank'].sum() for i in range(len(ranked_population))])

    ranked_population['probability'] = probability

    for round_i in range(2):

        in_round = np.array([np.random.choice(ranked_population['chromosome'],p=ranked_population['probability']) for i in range(tournament_size)])

        ranks = []

        for chrom in in_round:
            
            i = np.where(ranked_population['chromosome'] == chrom)[0][0]
            
            ranks.append(ranked_population['rank'].iloc[i])

        ranks = np.array(ranks)

        runners_up = in_round[ranks==max(ranks)]

        winner = np.random.choice(runners_up)

        parents.append(winner)
    
    return parents[0] , parents[1]

def main(seed, mutation_rate=0.01, crossover_rate=0.7, population_size=50, tournament_size=2):

    np.random.seed(seed)

    #This is essentially our termination condition.
    iterations = int(10000/population_size)

    population = generate_population(population_size)

    round_winners = []
    max_round_fitness = []
    min_round_fitness = []
    track_average_fitness = []


    for i in range(iterations):
        print("round: ",i)
        #Evaluate fitness and rank the population
        ranked_population = rank_population(population)
        print("fittest: ", ranked_population['fitness'].iloc[0])

        #Tracking fitness metrics
        fitnesses = np.array(ranked_population['fitness'])

        if i == 0:
            #On the first round there is nothing to compare to
            max_round_fitness.append((i,max(fitnesses)))
            min_round_fitness.append((i,min(fitnesses)))
        else:

            #Only want to record the fittest if it is newly the fittest otherwise we just end up with a whole list of duplicates telling us nothing.
            if max(fitnesses) != max_round_fitness[-1]:
                max_round_fitness.append((i,max(fitnesses)))
            if min(fitnesses) != min_round_fitness[-1]:
                min_round_fitness.append((i,min(fitnesses)))

            average_fitness = sum(fitnesses)/len(fitnesses)
            track_average_fitness.append(average_fitness)
            print("average fitness", average_fitness)

        #Select two parents for crossover
        parent_1, parent_2 = tournament_selection(ranked_population,tournament_size=tournament_size)

        #Crossover parents to create two new children with probability 0.7
        if np.random.rand(1)[0] <= crossover_rate:
            child_1 , child_2 = crossover(parent_1,parent_2)
            print("offspring produced")
        else:
            child_1 , child_2 = parent_1 , parent_2
            print("parents duplicated")

        #Mutate with probability mutation_rate. M is number of times to apply the mutation.
        M = np.random.randint(0,100)

        for j in range(M):
        
            if np.random.rand(1)[0] <= mutation_rate:
                child_1 = mutate(child_1)

        M = np.random.randint(0,100)

        for j in range(M):

            if np.random.rand(1)[0] <= mutation_rate: 
                child_2 = mutate(child_2)

        population = replacement(ranked_population,child_1,child_2)

        if i == 0:
            round_winners.append((i,population[0]))
        else:
            if round_winners[-1] != population[0]:
                round_winners.append((i,population[0]))

    with open(f'GA_run_{population_size}_{tournament_size}_{mutation_rate}_s{seed}.txt','w') as file:
        run_info = {
            'seed':seed,
            'iterations':iterations,
            'population_size':population_size,
            'tournament_size':tournament_size,
            'mutation_rate':mutation_rate,
            'crossover_rate':crossover_rate,
            'max_round_fitness':max_round_fitness,
            'min_round_fitness':min_round_fitness,
            'average_round_fitness':track_average_fitness,     
            'round_winners':round_winners
            }
        file.write(str(run_info))


    #fig, ax =plt.subplots(figsize=(16,8))
#
    #ax.plot(min_round_fitness, color='red',alpha=0.7)
    #ax.plot(max_round_fitness, color='green',alpha=0.7)
    #ax.plot(average_fitness, color='orange',alpha=0.7)
#
    #plt.show()

    return 0

if __name__=="__main__":

    """
    argv = 0 : run main as default
    argv = 1 : run with custom mutation and crossover
    """
    print(sys.argv)

    if sys.argv[1] == str(0):
        main(1)
    elif sys.argv[1]==str(1):
        mutation_rate=float(sys.argv[2])
        crossover_rate=float(sys.argv[3])
        population_size=int(sys.argv[4])
        tournament_size=int(sys.argv[5])
        seed=int(sys.argv[6])
        main(seed,mutation_rate,crossover_rate,population_size,tournament_size)
    else:
        print("None or invalid option. Terminating...")
        