# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from pacman import Directions
from game import Agent
from heuristics import *
import random
import math

class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

class RandomSequenceAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,10):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        possible = state.getAllPossibleActions();
        for i in range(0,len(self.actionList)):
            self.actionList[i] = possible[random.randint(0,len(possible)-1)];
        tempState = state;
        for i in range(0,len(self.actionList)):
            if tempState.isWin() + tempState.isLose() == 0:
                tempState = tempState.generatePacmanSuccessor(self.actionList[i]);
            else:
                break;
        # returns random action from all the valide actions
        return self.actionList[0];

class HillClimberAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        #initialize random sequence of size 5 with generic directions
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Hill Climber Algorithm instead of returning Directions.STOP  
        #get all possible actions for current state to create a random_action_list
        random_action_list = state.getAllPossibleActions()

        initial_sequence = []
        
        current_state = state

        #create initial random sequence
        for counter in range(0,5):
            new_random_action = random_action_list[random.randint(0,len(random_action_list)-1)]

            initial_sequence.append([new_random_action, current_state])
            
            #if winning state is found immediately return first action leading to it
            if current_state.isWin():
                return initial_sequence[0][0]
                
            elif current_state.isLose():
                break

            current_state = current_state.generatePacmanSuccessor(new_random_action)                     
        
        #determine overall performance of initial random sequence heuristic
        current_best_heuristic = gameEvaluation(initial_sequence[0][1], initial_sequence[-1][1])            
        
        current_state = state

        generator_exhausted = False
        
        new_sequence = initial_sequence

        #Runs until generator_exhausted == True
        while not generator_exhausted:

            #initializes empty new sequence    
            for counter in range(0,5):              

                if random.randint(1,100) <= 50:
                    new_sequence.append([initial_sequence[counter][0], current_state])
                    
                    new_state = current_state.generatePacmanSuccessor(new_sequence[counter][0])

                    #Check for winner
                    if new_state == None:
                        generator_exhausted = True
                        break
                    elif new_state.isWin():
                        return new_sequence[0][0]
                    elif new_state.isLose():
                        break

                    #current_state = current_state.generatePacmanSuccessor(new_sequence[counter][0])

                else:
                    new_random_action = random_action_list[random.randint(0,len(random_action_list)-1)]
                    new_sequence.append([new_random_action, current_state])

                    new_state = current_state.generatePacmanSuccessor(new_random_action)

                    #Check for winner
                    if new_state == None:
                        generator_exhausted = True
                        break
                    elif new_state.isWin():
                        return new_sequence[0][0]
                    elif new_state.isLose():
                        break

                    #current_state = current_state.generatePacmanSuccessor(new_random_action)

                current_state = new_state

            new_sequence_heuristic =  gameEvaluation(new_sequence[0][1], new_sequence[-1][1])

            if new_sequence_heuristic > current_best_heuristic:
                # assign new heuristic and update the current best sequence
                current_best_heuristic = new_sequence_heuristic
                initial_sequence = new_sequence

        return initial_sequence[0][0]
    
class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Genetic Algorithm instead of returning Directions.STOP

        #get all possible actions
        possible_actions_list = state.getAllPossibleActions()

        #generate initial population randomly
        current_population_container = self.generateInitialPopulation(possible_actions_list)

        forward_model_exhausted = False

        while not forward_model_exhausted:
            #get fitness rankings of each individual in the population
            curr_pop_fitness_rank, forward_model_exhausted = self.getFitnessOfEachPopulationMember(current_population_container, state)

            if not forward_model_exhausted:
                former_generation = current_population_container
                former_generation_fitness = curr_pop_fitness_rank
                current_population_container = self.breedPopulation(former_generation, former_generation_fitness, possible_actions_list)

        #Return the first index of the last complete generation by finding the index of the most fit individual 
        return former_generation[former_generation_fitness.index(max(former_generation_fitness))][0]

    def breedPopulation(self, generation_to_breed, generation_fitness, random_actions_list):
        #takes a generation and fitness array and breeds it with rankings
        #generate 8 individuals for the future population dependent upon the breeding rules
        
        new_population_generated = []

        sort_for_breeding = []

        for index,individual_fitness in enumerate(generation_fitness):
            sort_for_breeding.append([generation_to_breed[index],individual_fitness])

        sort_for_breeding.sort(key=lambda x:x[1], reverse = True)  

        parents_array = []
        new_child_list = []
        
        #using fit divisor we can sum throughj the population continuously to find parents to breed
        while len(new_population_generated) < 8:

            #get two parents as per the rules of fitness/gen_fitness_divisor liklihood
            for parent in range(0,2):
                
                #use uniform distribution to determine breeding liklihood of parents given the 8/26, 7/36, etc 
                #rules
                uni_distro = random.uniform(0.0, 36.0)

                if uni_distro < 8:
                    #8/36
                    parents_array.append(sort_for_breeding[0][0])
                elif uni_distro < 15:
                    #7/36
                    parents_array.append(sort_for_breeding[1][0])
                elif uni_distro < 21:
                    #6/36
                    parents_array.append(sort_for_breeding[2][0])

                elif uni_distro < 26:
                    #5/36
                    parents_array.append(sort_for_breeding[3][0])

                elif uni_distro < 30:
                    #4/36
                    parents_array.append(sort_for_breeding[4][0])

                elif uni_distro < 33:
                    #3/36
                    parents_array.append(sort_for_breeding[5][0])

                elif uni_distro < 35:
                    #2/36
                    parents_array.append(sort_for_breeding[6][0])

                elif uni_distro < 36:
                    #1/36
                    parents_array.append(sort_for_breeding[7][0])
        
            #individual parents
            parent_x = parents_array[0]
            parent_y = parents_array[1]
            
            #Check to see if parents will generate two children by crossing over?
            #generate two children by crossing-over
            if random.randint(0,100) <= 70:
                
                for new_child in range(0,2):
                    
                    for genes in range(len(parent_x)):

                        if random.randint(0,100) < 50:
                            new_child_list.append(parent_x[genes])
                        else:
                            new_child_list.append(parent_y[genes])

                    new_population_generated.append(new_child_list)
                    
                    new_child_list = []
            else:
                #30% probability of both parents remaining within the population
                new_population_generated.append(parent_x)
                new_population_generated.append(parent_y)       

        #mutate random gene based upon 10% likelihood of mutation
        for individual in new_population_generated:    
            #mutate randomly 
            if random.randint(0,100) <= 10:
                individual[random.randint(0, len(individual)-1)] = random_actions_list[random.randint(0, len(random_actions_list)-1)]

        #return new_population_generated
        return new_population_generated

    def generateInitialPopulation(self, random_action_list):
        #Generates initial population randomly by creating 8 random 5 chromosome sequences
        #Returns in a list
        population_container = []

        #Create an initial population of 8 with 5 chromosomes each (in this case chromosome indicates sequence direction)
        for counter in range(0,8):
    
            #generate random chromosomes for starting population
            population_container.append([random.randint(0,len(random_action_list)-1), 
                random.randint(0,len(random_action_list)-1),random.randint(0,len(random_action_list)-1), 
                random.randint(0,len(random_action_list)-1),random.randint(0,len(random_action_list)-1)])

        return population_container
    
    def getFitnessOfEachPopulationMember(self, population_container, seed_state):
        #Generates an array of fitness measurements for each member of the population
        #Array should have same indexes as population
        population_fitness_array = [] 

        forward_model_exhausted = False       

        for counter in range(len(population_container)):

            current_state = seed_state

            current_individual = population_container[counter]

            for inner_counter in range(len(current_individual)):
                current_state = current_state.generatePacmanSuccessor(current_individual[inner_counter])

                if current_state == None:
                    forward_model_exhausted = True
                    return population_fitness_array, forward_model_exhausted
                elif current_state.isWin():
                    foundWinner = True
                    break
                elif current_state.isLose():
                    break

            if not forward_model_exhausted or foundWinner:
                population_fitness_array.append(gameEvaluation(seed_state, current_state))

        return population_fitness_array, forward_model_exhausted

class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.C_VALUE = 1
        self.treeRoot = None
        return;

    class treeNode:
        
        def __init__(self, parent, direction_to_here):
            self.move_direction = direction_to_here
            self.parent = parent
            self.children = []
            self.visits = 0
            self.score = 0            
            self.wins = 0

        def addChild(self, new_child):
            self.children.append(new_child)

        def getParent(self):
            return self.parent
        
        def getChildren(self):
            return self.children
        
        def incrementWins(self):
            self.wins += 1
        
        def visited(self):
            self.visits += 1

        def updateScore(self, new_score):
            self.score = (self.score + new_score)/self.visits

        def getScore(self):
            return self.score

    # GetAction Function: Called with every frame
    def getAction(self, state):
        #1) Selection  UCB1 (Si) = Vi + C Sqrt(Ln(N)/ni)   C = 1
        #2) Node Expansion
        #3) Rollout (random simulation)
        #4) Backpropagation
        # determine node by most visited node

        current_state = state

        forward_model_exhausted = False

        while not forward_model_exhausted:
            #Selection Process
            selected_node = self.Selection(current_state)

            #Expansion Process
            state_for_rollout, forward_model_exhausted = self.Expansion(selected_node, current_state)
            
            #Rollout exercise
            if not forward_model_exhausted:
                rollout_results, forward_model_exhausted = self.Rollout(state_for_rollout)

            if not forward_model_exhausted:
                self.BackpropagateResults(rollout_results, selected_node)

        #find most visisted node
        return self.FindMostVisistedNode() 
    
    def FindMostVisistedNode(self):
        #Find the most visited node 
        current_node_to_check = self.treeRoot
        
        child_list = current_node_to_check.getChildren()

        current_most_visited_node = child_list[0]
        num_of_visits = child_list[0].visits

        while child_list != []:

            current_node_to_check = child_list.pop(0)

            new_children = current_node_to_check.getChildren()
            
            for child in new_children:
                child_list.append(child)

            if current_node_to_check.visits > num_of_visits:
                num_of_visits = current_node_to_check.visits
                current_most_visited_node = current_node_to_check
        
        find_direction_node = current_most_visited_node

        while find_direction_node.getParent().getParent() != None:

            find_direction_node = find_direction_node.getParent()
        

        return find_direction_node.move_direction
    
    def getToCurrentState(self, action_list, current_state):
        #returns appropriate state for rollout to avoid unecessary iterative rollouts for the same process
        real_current_state = current_state
        forward_model_exhausted = False

        #move state forward to current position on tree for expansion
        while not len(action_list) == 0 and real_current_state is not None:
            previous_state = real_current_state

            real_current_state = real_current_state.generatePacmanSuccessor(action_list.pop())

            if real_current_state is None:
                forward_model_exhausted = True
                real_current_state = previous_state
                break

            if real_current_state.isWin():
                break
            elif real_current_state.isLose():
                break

        return real_current_state, forward_model_exhausted

    def BackpropagateResults(self, rollout_results, selected_node):
        #Take results and backpropagate back up the tree
        #list structure [score_of_rollout, win_found]
        
        continue_to_root = True
        current_node = selected_node

        while continue_to_root:
            current_node.visited()
            current_node.updateScore(rollout_results[0])
            
            if rollout_results[1]:
                current_node.incrementWins()

            if not current_node.getParent() is None:
                current_node = current_node.getParent()
            else:
                continue_to_root = False


    def Rollout(self,state_for_rollout):
        #conducts 5 state forward rollout exercise
        #returns a list of rollout statistics for back propagation and checks forward_model exhaustion
        score_of_rollout = 0

        win_found = False

        simulated_state = state_for_rollout

        forward_model_exhausted = False

        for counter in range(0,6):
            #extra counter to account for checks on initial state

            if simulated_state is None:
                forward_model_exhausted = True
                break
            if simulated_state.isWin():
                win_found = True
                break
            elif simulated_state.isLose():
                break
            else:
                current_legal_actions = simulated_state.getLegalPacmanActions()
                simulated_state = simulated_state.generatePacmanSuccessor(current_legal_actions[random.randint(0,len(current_legal_actions)-1)])
            
        if not simulated_state is None:
            score_of_rollout = gameEvaluation(state_for_rollout, simulated_state)

        return [score_of_rollout, win_found], forward_model_exhausted

    def Expansion(self, selected_node, current_state):
        #Expansion phase of the model
        #if node never visited before...immediately roll out
        #if node has been visited...expand then roll out
        
        all_actions_to_node = self.getAllActionsToThisNode(selected_node)
        
        if not selected_node.visits == 0:
            #node has been visisted...expand it
            state_for_rollout, forward_model_exhausted = self.getToCurrentState(all_actions_to_node, current_state)

            if forward_model_exhausted:
                forward_model_exhausted = True
                return current_state, forward_model_exhausted

            legal_action_list_in_state = state_for_rollout.getLegalPacmanActions()

            for legal_action in legal_action_list_in_state:
                selected_node.addChild(self.treeNode(selected_node, legal_action))
        else:
            #node never visisted...immediately roll it out
            state_for_rollout, forward_model_exhausted = self.getToCurrentState(all_actions_to_node, current_state)

        return state_for_rollout, forward_model_exhausted

    def Selection(self, current_state):
        #start by checking if treeRoot is none
        #complete selection process for the node
        
        if not self.treeRoot == None:
            continue_searching = True
            current_node = self.treeRoot

            UCB1_Score_Container = []

            while continue_searching:
                #return children
                if current_node.getChildren() != []:
                    curr_node_children = current_node.getChildren()

                    for child in curr_node_children:
                        
                        if child.visits == 0:
                            #Infinity value found...will automatically be selected
                            child_selected = child
                            continue_searching = False
                            return child_selected 
                            
                        else:
                            UCB1_Score_Container.append(child.getScore() + self.C_VALUE * math.sqrt((math.log(current_node.visits)/child.visits)))
                    
                    #find node with highest UCB1 score and start checking that one out
                    if not all(score==UCB1_Score_Container[0] for score in UCB1_Score_Container):
                        current_node = curr_node_children[UCB1_Score_Container.index(max(UCB1_Score_Container))]
                    else:
                        current_node = curr_node_children[random.randint(0,len(curr_node_children)-1)]
                        
                    UCB1_Score_Container[:] = []
                else:
                    # leaf node found...should return
                    continue_searching = False
                    child_selected = current_node
        else:
            #generate legal actions and create first node
            #initially all unvisited nodes in the tree will have an infinite score so choice of first action
            #is random
            #initial node simply being set to an empty node that will act as the initial state
            #unsure if this is correct, but thinking of this as the very top of the tree
            self.treeRoot = self.treeNode(None,None)
            current_legal_actions = current_state.getLegalPacmanActions()
            
            for child in current_legal_actions:
                #add children to treeRoot node
                self.treeRoot.addChild(self.treeNode(self.treeRoot, child))

            return self.treeRoot.getChildren()[0]

        return child_selected
    
    def getAllActionsToThisNode(self, selected_node):
        #gets all actions to this node to generate forward state for expansion
        #creates sequence to arrive at this point on the tree
        #must be treated as LIFO iterating back up the tree
        action_list_to_current = []
        
        current_node = selected_node

        while current_node.getParent() is not None:
            action_list_to_current.append(current_node.move_direction)
            current_node = current_node.getParent()

        return action_list_to_current