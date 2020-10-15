"""
    Programmer: Khalid Hassan
     
    Paper 1: A New Heuristic Optimization Algorithm: Harmony Search
    Authors: Zong Woo Geem and Joong Hoon Kim, G. V. Loganathan

    Paper 2: An improved harmony search algorithm for solving optimization problems
    Authors: M. Mahdavi, M. Fesanghary, E. Damangir 
"""

import numpy as np
import matplotlib.pyplot as plt
import math, time, sys, random

from sklearn.model_selection import train_test_split
from sklearn import datasets

from _utilities import Solution, Data, initialize, sort_agents, display, compute_accuracy

def HS(num_agents, max_iter, train_data, train_label, obj_function = compute_accuracy, HMCR = 0.90, save_conv_graph = False):
    # Harmony Search Algorithm
    # Parameters
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: number of harmonies                                           #
    #   max_iter: maximum number of generations                                   #
    #   train_data: training samples of data                                      #
    #   train_label: class labels for the training samples                        #                
    #   obj_function: the function to maximize while doing feature selection      #
    #   HMCR: Harmony Memory Consideration Rate                                   #
    ############################################################################### 


    # <STEPS OF HARMOMY SEARCH ALGORITH>
    # Step 1. Initialize a Harmony Memory (HM).
    # Step 2. Improvise a new harmony from HM.
    # Step 3. If the new harmony is better than minimum harmony in HM, include the new harmony in HM, and exclude the minimum harmony from HM.
    # Step 4. If stopping criteria are not satisfied, go to Step 2.


    train_data, train_label = np.array(train_data), np.array(train_label)
    num_features = train_data.shape[1]

    short_name = "HS"
    agent_name = "Harmony Search"

    ########################################                 STEP: 1 Initialize                                   ########################################
    # Intialisation
    harmonyMemory = initialize(num_agents, num_features);
    fitness = np.zeros(num_agents)
    Leader_agent = np.zeros((1, num_features))
    Leader_fitness = float("-inf")

    # initialize convergence curves
    convergence_curve = {}
    convergence_curve['fitness'] = np.zeros(max_iter)
    convergence_curve['feature_count'] = np.zeros(max_iter)

    # initialise data class
    data = Data()
    data.train_X, data.val_X, data.train_Y, data.val_Y = train_test_split(train_data, train_label, stratify=train_label, test_size=0.2)

    # Initialise solution class
    solution = Solution()
    solution.num_agents = num_agents
    solution.max_iter = max_iter
    solution.num_features = num_features
    solution.obj_function = obj_function

    # start timer
    start_time = time.time()

    # calculate initial fitess and sort the harmony memory and rank them
    harmonyMemory, fitness = sort_agents(harmonyMemory, obj_function, data)

    # Create new harmonies in each iteration
    for iterCount in range(max_iter):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(iterCount + 1))
        print('================================================================================\n')
        HMCR_randValue = np.random.rand()
        newHarmony = np.zeros([1, num_features])

        print(HMCR)
        print(HMCR_randValue)

        ########################################                 STEP: 2 Improvise Harmony                                 ########################################
        if HMCR_randValue <= HMCR:
            for featureNum in range(num_features):
                selectedAgent = random.randint(0, num_agents - 1)

                newHarmony[0, featureNum] = harmonyMemory[selectedAgent, featureNum]

        else:
            for featureNum in range(num_features):
                newHarmony[0, featureNum] = random.randint(0, 1)

        ########################################                 STEP: 3 Replace better performing harmony                 ########################################
        fitnessHarmony = obj_function(newHarmony, data.train_X, data.val_X, data.train_Y, data.val_Y)

        if fitness[num_agents-1] < fitnessHarmony:
            harmonyMemory[num_agents-1, :] = newHarmony
            fitness[num_agents-1] = fitnessHarmony

        # Sort harmony memory
        harmonyMemory, fitness = sort_agents(harmonyMemory, obj_function, data)


        # Update 
        convergence_curve['fitness'][iterCount] = fitness[0];
        convergence_curve['feature_count'][iterCount] = int(np.sum(harmonyMemory[0, :]))

        display(harmonyMemory, fitness, agent_name)
    
    # leader agent and leader fitneess
    Leader_fitness = fitness[0]
    Leader_agent = harmonyMemory[0].copy();


    # stop timer
    end_time = time.time()
    exec_time = end_time - start_time

    # plot convergence curves
    iters = np.arange(max_iter)+1
    fig, axes = plt.subplots(2, 1)
    fig.tight_layout(pad=5)
    fig.suptitle('Convergence Curves')

    axes[0].set_title('Convergence of Fitness over Iterations')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Fitness')
    axes[0].plot(iters, convergence_curve['fitness'])

    axes[1].set_title('Convergence of Feature Count over Iterations')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Number of Selected Features')
    axes[1].plot(iters, convergence_curve['feature_count'])

    if(save_conv_graph):
        plt.savefig('convergence_graph_'+ short_name + '.png')
    plt.show()


    solution.best_agent = Leader_agent
    solution.best_fitness = Leader_fitness
    solution.convergence_curve = convergence_curve
    solution.final_population = harmonyMemory
    solution.final_fitness = fitness
    solution.execution_time = exec_time


    return solution


if __name__ == '__main__':

    iris = datasets.load_iris()
    HS(10, 20, iris.data, iris.target, compute_accuracy, save_conv_graph=True)





    



        

