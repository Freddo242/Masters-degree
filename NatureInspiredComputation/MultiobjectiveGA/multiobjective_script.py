import numpy as np
import onemax_script
import csv
import os

from tqdm import tqdm

from collections import Counter

def generate_population(size):
    """
    A chromosome will be [pop_size,t_size,m_rate,m_size,cross_rate
    pop_size between 5 and 20
    t_size between 2 and 10
    m_rate between 0 and 1
    m_size between 0 and 15
    crossover rate between 0 and 1
    """
    return [ [np.random.randint(5,20), 2*np.random.randint(1,5), np.random.rand(),np.random.randint(0,15),np.random.rand()] for i in range(size) ]

def fitness(individual,it):

    #fitness of an individual is 1000-mean_num_iterations. 
    iterations = _main_function_.run_ga(individual[0],individual[1],individual[2],individual[3],individual[4],it)
    #Taking absolute value so we don't end up with negatives ranking higher than the rest
    mean = abs(1000 - np.mean(iterations))
    std = np.std(iterations)

    if std == 0.0:
        std = 900

    return (mean, std)
    
def assign_ranks(population_fitness):
    #Takes a tuple of (chromosome,mean,std) sorted by mean. returns (rank,chromosome,mean,std)
    ranked_pop = [] 
    rank = 1

    while True:
        #indexes of the candidates we'll assign this rank to
        pareto_front= []
        compc = population_fitness[0]
        pareto_front.append(0)
        ranked_pop.append( (rank,compc[0],compc[1],compc[2]) )
        
        #finding candidates also in this pareto front
        for i in range(1,len(population_fitness)):
            candidate = population_fitness[i]
            if candidate[2] < compc[2]:
                compc = candidate
                pareto_front.append( i )
                ranked_pop.append( (rank,candidate[0],candidate[1],candidate[2]) )

        #these have been added to 
        pareto_front.sort(reverse=True)
        for i in pareto_front:
            population_fitness.pop(i)

        if len(population_fitness) != 0:
            #Now calculate the next pareto front
            rank += 1
            compc = population_fitness[0]
        else:
            break
    
    return ranked_pop

def tournament(ranked_population,t_size=2):
    #selecting parents based on tournament selection with no replacement

    if t_size >= len(ranked_population):
        print("tournament size greater than or equal to population size")
        return 0

    #creating the probabilities of being selected
    ranks = list(np.array(ranked_population,dtype=object)[:,0])
    c = Counter(ranks)
    ranks = list(set(ranks))
    rank_sum = sum(ranks)
    rank_prob = { ranks[i] : ranks[i-1]/rank_sum  for i in range(len(ranks)) }

    #There are a lot of subltities here because candidates can share ranks and therefore probabilities
    probabilities = [ rank_prob[ranked_population[i][0]]/c[ranked_population[i][0]] for i in range(len(ranked_population))]

    parents = []

    for r in range(2):
        tourn = []
        pool = ranked_population.copy()

        #picks random index based on rank probabilities and pops it, adding it to the tournament
        to_pop = np.random.choice(np.arange(len(pool)),size=t_size, replace=False, p=probabilities)
        to_pop = np.sort(to_pop)[::-1]

        for index in to_pop:
            tourn.append( pool.pop(index) )
        
        #Pick a candidate to compare the others to from what is left in the pool
        compc = pool[np.random.randint(len(pool))]

        final = []

        for candidate in tourn:
            if candidate[0] > compc[0]:
                final.append(candidate)

        if len(final) > 1:
            #split the tie randomly between winners
            parents.append(final[np.random.randint(len(final))])
        elif len(final) == 1:
            #We have only one winner
            parents.append(final[0])
        else:
            #We have no winners, so take randomly from those in the tournament
            parents.append(tourn[np.random.randint(len(tourn))])                

    return parents

def tournament_biased(ranked_population,t_size=2):
    #selecting parents based on tournament selection with no replacement

    if t_size >= len(ranked_population):
        print("tournament size greater than or equal to population size")
        return 0

    #creating the probabilities of being selected
    ranks = list(np.array(ranked_population,dtype=object)[:,0])
    c = Counter(ranks)
    ranks = list(set(ranks))
    rank_sum = sum(ranks)
    rank_prob = { ranks[i] : ranks[i-1]/rank_sum  for i in range(len(ranks)) }

    #There are a lot of subltities here because candidates can share ranks and therefore probabilities
    probabilities = [ rank_prob[ranked_population[i][0]]/c[ranked_population[i][0]] for i in range(len(ranked_population))]

    parents = []

    pool = ranked_population.copy()
    #picks random index based on rank probabilities and pops it, adding it to the tournament
    indexes = np.random.choice(np.arange(len(pool)),size=t_size, replace=False, p=probabilities)
    tourn = [pool[i] for i in indexes]
    stds = np.array([ chrom[-1] for chrom in tourn ])
    index = np.where(stds == min(stds))[0][0]
    parents.append(tourn[index])

    pool = ranked_population.copy()
    #picks random index based on rank probabilities and pops it, adding it to the tournament
    indexes = np.random.choice(np.arange(len(pool)),size=t_size, replace=False, p=probabilities)
    tourn = [pool[i] for i in indexes]
    means = np.array([ chrom[-2] for chrom in tourn ])
    index = np.where(means == min(means))[0][0]
    parents.append(tourn[index])

    return parents


def crossover(parent_1,parent_2,c_rate):

    parent_1 = parent_1[1]
    parent_2 = parent_2[1]

    rn = np.random.randn(1)[0]
    if rn < c_rate:
        index = np.random.randint(len(parent_1))
        return parent_1[:index] + parent_2[index:] , parent_2[:index] + parent_1[index:]
    else:
        return parent_1, parent_2

def mutate(child,m_rate,m_size):
    #Instead of the mutation size being

    index = np.random.randint(len(child))
    if np.random.rand(1)[0] < m_rate:
        
        if index == 0:
            #Either increase or decrease the population size by m_size
            new = int(child[0]*(1+m_size*[-1,1][np.random.randint(2)]))
            if new < 5:
                new = 5
            if new > 20:
                new = 20
            child[0] = new

        elif index == 1:
            #mutate t_size up or down 2
            new = child[1] + [-2,2][np.random.randint(2)]
            if new > 10: 
                new = 10
            if new < 2:
                new = 2
            if new > child[0]:
                new = 2
            child[1] = new

        elif index==2:
            #mutation rate
            new = child[2]*(1+m_size*[-1,1][np.random.randint(2)])
            if new <= 0:
                new = 0.01
            if new > 1:
                new = 0.99
            child[2] = new

        elif index==3:
            #mutation size
            new = child[3]*(1+m_size*[-1,1][np.random.randint(2)])
            if new <= 0:
                new = 1
            if new > 15:
                new = 15
            new = child[3]

        elif index == 4:
            #mutate crossover rate
            new = child[4]*(1+m_size*[-1,1][np.random.randint(2)])
            if new <= 0:
                new = 0.01
            if new > 1:
                new = 0.99
            child[4] = new

        else:
            print("error here in mutation function.")

    return child


def replacement(ranked_population,c1,c2):

    #Accepts ranked population with two chromosomes to insert. Returns population as list of tuples (chromosome,mean,std)
    c1_fit = fitness(c1,50)
    c2_fit = fitness(c2,50)

    #print("children ",c1_fit,c2_fit)

    #strip population of rank
    rp = [cand[1:] for cand in ranked_population]

    #add children 
    rp.append((c1,c1_fit[0],c1_fit[1]))
    rp.append((c2,c2_fit[0],c2_fit[1]))

    ranked = assign_ranks(rp)
    ranks = {}
    for cand in ranked:
        try:
            ranks[cand[0]].append(cand)
        except KeyError:
            ranks[cand[0]] = [cand]

    ranks = { key: ranks[key] for key in sorted(ranks.keys()) }

    #removes random candidates from the lowest ranks
    for i in range(2):
        r = list(ranks.keys())
        if len(ranks[r[-1]]) > 0:
            ranks[r[-1]].pop(np.random.randint(len(ranks[r[-1]])))
        else:
            ranks[r[-2]].pop(np.random.randint(len(ranks[r[-2]])))

    new_pop = []
    for r in list(ranks.values()):
        new_pop = new_pop + r

    return [cand[1:] for cand in new_pop]
        
def main(p_size,t_size,c_rate,m_rate,m_size):

    iterations = 500

    infile = open(f'run_{p_size}_{t_size}_{c_rate}_{m_rate}_{m_size}.csv','w')
    writer = csv.writer(infile)

    population = generate_population(p_size)

    population_fitness = []
    for candidate in population:
        fit = fitness(candidate,50)
        population_fitness.append( (candidate,fit[0],fit[1]) )

    #sorting by the mean ascending
    population_fitness.sort(key=lambda a:a[1])

    for i in tqdm(range(iterations)):
        #We should only have to calculate the fitness of the children each round rather than the whole population.

        #print(i)
        ranked_population = assign_ranks(population_fitness)

        #parents = tournament(ranked_population,t_size=t_size)
        parents = tournament_biased(ranked_population,t_size=t_size)

        child_1 , child_2 = crossover(parents[0],parents[1],c_rate)

        child_1 = mutate(child_1,m_rate,m_size)
        child_2 = mutate(child_2,m_rate,m_size)

        population_fitness = replacement(ranked_population, child_1,child_2)

        row = [str(i)]
        for cand in population_fitness:
            row = row + [str(cand[1]), str(cand[2]), ' ']
        row = row + [str(population_fitness[0][0])]

        writer.writerow(row)

    #for ind in population:
    #   print(ind, fitness(ind))

    infile.close()

if __name__ == "__main__":

    #population size
    p_size = 50
    #tournament size
    t_size = 2
    #crossover rate
    c_rate = 0.9
    #mutation rate
    m_rate = 0.2
    #mutation size is a percentage of how much we mutate the param by
    m_size = 0.1

    main(p_size,t_size,c_rate,m_rate,m_size)


