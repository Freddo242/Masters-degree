# Multi-Objective EA README

## General overview

The Multi-Objective EA (MOEA) genetically evolves chromosomes representing the parameters for a 1-MAX genetic algorithm. The MOEA attempts to minimise the mean and standard deviation of the number of iterations it took the 1-MAX algorithm, with a chromosome's parameters, over 50 iterations.

The file includes the MOea.py script, and the _main_function_.py script (which runs the 1-MAX EA)


## Modules

numpy
csv
tqdm

## How to use

The script takes no command line arguments
To run the MOEA, change the parameters in the main() function, which are the parameters for the MOEA, and then run the script.
It will create a file to record the mean, standard deviation, and fittest chromosome for each round.

## Functions

Fitness - calculates the mean and standard deviation of the iteration which the 1-MAX algorithm failed on.

assign_ranks - takes in a list of tuples (chromosome, mean, std), and outputs tuples with their ranks (rank,chromosome,mean,std) as ranked_population.

tournament - takes in ranked_population and outputs two parents based on the probabilities weighted by the ranks of the chromosomes.

tournament_biased - input and output is the same as tournament function, but it chooses the parents based on which chromosome in the tournament has a smaller standard deviation

crossover - takes in two chromosome and outputs two child chromosomes 

mutate - mutates a chromosome depending on mutation rate and mutation size. The mutation size is a percentage which relates to how much to mutate a gene.

replacement - takes in two chromosomes and a ranked population, ranks them all, and removes two chromosomes from the lowest rank


