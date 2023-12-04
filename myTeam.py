# baselineTeam.py
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random, time, util, sys
from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint
import math
from distanceCalculator import manhattanDistance


# DIGITAL DRAGONS SUBMISSION

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among legal actions randomly. 
        """
        actions = game_state.get_legal_actions(self.index)

        return random.choice(actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

    
class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food using MCTS
  """
    def register_initial_state(self, game_state):
        """
        Registers the initial state of the game
        """
        
        CaptureAgent.register_initial_state(self, game_state)

        self.initialPosition = game_state.get_initial_agent_position(self.index)

        # Calculate the escape hatches: the entry points along the middle line of the gamespace.
        # Red on left, Blue on right

        self.escapeHatch = []
        if self.red:
            # print("I'M RED TEAM")
            # Centre is on the left hand side of the middle line
            centre_width = int((game_state.data.layout.width - 2) / 2)

        else:
            # print("I'M BLUE TEAM")
            # Centre is on the right hand side of the middle line
            centre_width = int(((game_state.data.layout.width -2) / 2) + 1)

        # Finding the "gaps" in the maze as escape hatches
        for height in range(1, game_state.data.layout.height-1):
            if not game_state.has_wall(centre_width, height):
                self.escapeHatch.append((centre_width, height))

        self.move_count = 0

        self.start = game_state.get_agent_position(self.index)

        self.safe_space = self.start 

        self.entryPoints = self.escapeHatch

        self.survival_mode = False

        self.power_mode = False

        food_grid = self.get_food(game_state)
        food_positions = []
        
        # Generating list of pellet coordinates
        for x in range(game_state.data.layout.width):  # Iterate over each row
            for y in range(game_state.data.layout.height):  # Iterate over each column in the row
                if food_grid[x][y]:  # Check if there's food at position (x, y)
                    food_positions.append((x, y))  # Append the position to the list

        self.totalFood = len(food_positions)
        self.isOpRed = not self.red

        # Find starting positions of opponents in the maze
        x,y = 0,0
        if self.isOpRed:
            x = 1
            y = 1
        else:
            x = game_state.data.layout.width - 2
            y = game_state.data.layout.height - 2

        self.ghost_pos = (x,y)

    def choose_action(self, game_state):
        """
        Chooses the best action using MCTS techniques combined with goal
        recognition, evaluating different states of the game.
        """
        # Food our agent is carrying
        foodCarrying = game_state.get_agent_state(self.index).num_carrying

        self.survival_mode_check(game_state)
        self.power_check(game_state)
        if not game_state.get_agent_state(self.index).is_pacman:
            self.safe_space = self.start

        self.actionStartTime = time.time()

        # Iterations of Monte Carlo Tree
        # Simulation Depth of Monte Carlo
        self.CONST_ITERATION_TIMES = 32
        self.CONST_MONTE_CARLO_DEPTH = 10

        # If in survival mode or carrying lots of food, flee, otherwise determine best action 
        if self.survival_mode is False and foodCarrying < 6:
            MCT = self.MCT(game_state, iter_times=self.CONST_ITERATION_TIMES,\
                                        simulate_depth=self.CONST_MONTE_CARLO_DEPTH)

            children = MCT.children
            values = [child.reward/child.visits for child in children]
            max_value = max(values)
            nextState = [c for c, v in zip(children, values) if v == max_value][0]

            # print( 'eval time for agent %d: %.4f' % (self.index, time.time() - self.actionStartTime))
            self.move_count += 1
            
            action = nextState.game_state.get_agent_state(self.index).configuration.direction
            if self.power_mode is True:
                print("Features: ", self.get_features(game_state, action))
                print("weights`; ", self.get_weights(game_state, action))
            return action
        else:
            action = self.go_home(game_state)

            # print( 'eval time for agent %d: %.4f' % (self.index, time.time() - self.actionStartTime))
            if action is None:
                actions = game_state.get_legal_actions(self.index)
                self.move_count += 1
                return random.choice(actions)
            else:
                self.move_count += 1
                return action

    def check_dead_end(self, game_state, action, depth):
        """
        Recursively checks if following a sequence of actions leads to a dead end.
        Args:
            game_state: The current state of the game.
            action: The action to check for leading to a dead end.
            depth: The depth to which the check should be performed.
        Returns:
            True if the sequence of actions leads to a dead end, False otherwise.
        """
        # Base case: If reached the required depth, return False (not a dead end)
        if depth == 0:
            return False

        # Generate the successor state after performing the action
        successor = game_state.generate_successor(self.index, action)
        actions = successor.get_legal_actions(self.index)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)

        # Remove the reverse direction from possible actions to avoid oscillation
        currentDirection = successor.get_agent_state(self.index).configuration.direction
        reverse_direction = Directions.REVERSE[currentDirection]
        if reverse_direction in actions:
            actions.remove(reverse_direction)

        # If no actions are possible, it's a dead end
        if len(actions) == 0:
            return True

        # Recursively check each remaining action
        for next_action in actions:
            if not self.check_dead_end(successor, next_action, depth - 1):
                return False  # Found a path that is not a dead end

        # If all paths lead to a dead end, return True
        return True
    
    def check_empty_path(self, game_state, action, depth):
        """
        Recursively checks if a path leads to a dead end without any food.
        Args:
            game_state: The current state of the game.
            action: The action to check.
            depth: The depth to which the check should be performed.
        Returns:
            True if the path is empty (leads to a dead end with no food), False otherwise.
        """
        if depth == 0:
            return False

        successor = game_state.generate_successor(self.index, action)
        score = game_state.get_agent_state(self.index).num_carrying
        new_score = successor.get_agent_state(self.index).num_carrying

        # If the position contains a capsule or the score increases, it's not an empty path
        my_pos = successor.get_agent_position(self.index)
        if my_pos in self.get_capsules(game_state) or score < new_score:
            return False

        # Check legal actions from the successor state, excluding STOP and reverse direction
        actions = successor.get_legal_actions(self.index)
        actions.remove(Directions.STOP)
        reverse_direction = Directions.REVERSE[successor.get_agent_state(self.index).configuration.direction]
        if reverse_direction in actions:
            actions.remove(reverse_direction)

        # If no actions are possible, it's an empty path
        if len(actions) == 0:
            return True

        # Recursively check each remaining action
        for next_action in actions:
            if not self.check_empty_path(successor, next_action, depth - 1):
                return False  # Found a path that is not empty

        # All paths lead to dead ends without food
        return True

    def go_to_capsule(self, game_state):
        """
        Determines the best action to take to move towards capsules while avoiding ghosts.
        Args:
            game_state: The current state of the game.
        Returns:
            The best action to move towards a capsule while avoiding ghosts.
        """
        actions = game_state.get_legal_actions(self.index)
        best_dist = float('inf')  
        best_action = None

        # Get positions of ghosts that are not scared
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghost_pos = [a.get_position() for a in enemies if not a.is_pacman and a.get_position() != None and a.scared_timer == 0]

        # Filter out actions that lead to dead ends or unsafe paths
        actions = [action for action in actions if not self.check_empty_path(game_state, action, 20)]

        for action in actions:
            successor = self.get_successor(game_state, action)
            pos2 = successor.get_agent_position(self.index)
            capsules = self.get_capsules(successor)

            # Check if a capsule is collected
            if len(self.get_capsules(game_state)) > len(capsules):
                return action

            # Find the nearest capsule in the successor state
            if capsules:
                min_dist, minCap = min((self.get_maze_distance(pos2, cap), cap) for cap in capsules)

                # Increase distance if close to ghosts to avoid them
                if any(self.get_maze_distance(pos2, gp) < 2 for gp in ghost_pos):
                    min_dist += 99999999

                # Update the best action if this is the closest we've found to a capsule
                if min_dist < best_dist:
                    best_action = action
                    best_dist = min_dist

        return best_action

 
        # TODO: Implement MCTS logic for returning home safely
        pass
   
    def go_home(self, game_state):
        """
        Determines the best action to return home safely, considering capsules and ghosts.
        Args:
            game_state: The current state of the game.
        Returns:
            The best action to return home while avoiding ghosts or going for a capsule if safe.
        """
        actions = game_state.get_legal_actions(self.index)
        best_dist = float('inf')  
        best_action = None

        # Identify positions of ghosts that are not scared
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghost_pos = [a.get_position() for a in enemies if not a.is_pacman and a.get_position() != None and a.scared_timer < 5]

        # Check if the agent is closer to any capsule than the ghosts
        capsuleList = self.get_capsules(game_state)
        if capsuleList:
            for cap in capsuleList:
                ghostToCap = None
                valid_ghost_positions = [gp for gp in ghost_pos if gp is not None]
                if valid_ghost_positions:
                    ghostToCap = min(self.get_maze_distance(cap, gp) for gp in valid_ghost_positions)

                disToCap = self.get_maze_distance(cap, game_state.get_agent_position(self.index))
                if disToCap < ghostToCap:
                    return self.go_to_capsule(game_state)

        # Filter out actions leading to dead ends
        actions = [action for action in actions if not self.check_dead_end(game_state, action, 20)]

        # Evaluate each action to find the best one to go back home
        for action in actions:
            successor = self.get_successor(game_state, action)
            pos2 = successor.get_agent_position(self.index)
            dist = self.get_maze_distance(self.safe_space, pos2)

            # Avoid actions that lead too close to ghosts
            if any(pos2 == gp or self.get_maze_distance(pos2, gp) < 2 for gp in ghost_pos):
                dist += 99999999

            # Update the best action if this is the closest we've found to the survival point
            if dist < best_dist:
                best_action = action
                best_dist = dist

        return best_action

    def survival_mode_check(self, game_state):
        """
        Checks various conditions to determine if the agent should enter survival mode.
        Args:
            game_state: The current state of the game.
        Sets:
            self.survival_mode: Boolean indicating if the agent is in survival mode.
            self.safe_space: The target point for survival mode.
        """
        self.survival_mode = False
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        # Identify nearby ghosts that are not scared
        chase_ghost = [a for a in enemies if not a.is_pacman and a.get_position() != None and a.scared_timer < 5]
        myState = game_state.get_agent_state(self.index)
        my_pos = myState.get_position()

        gDist = 9999
        gMDist = 9999
        if len(chase_ghost) > 0:
            # Calculate distance to the nearest ghost
            gDist = min([self.get_maze_distance(my_pos, cg.get_position()) for cg in chase_ghost])
            gMDist = min([manhattanDistance(my_pos, cg.get_position()) for cg in chase_ghost])

        food_carrying = myState.num_carrying  # Food currently carried by the agent
        food_left = 0
        food_grid = self.get_food(game_state)  # Remaining food in the game
        # Iterate over the grid to count food items
        for x in range(food_grid.width):
            for y in range(food_grid.height):
                if food_grid[x][y]:  # Assuming this checks for food at the (x, y) position
                    food_left += 1
        # Check various conditions to activate survival mode
        if myState.is_pacman and food_carrying > (self.totalFood / 2 - 1) and len(chase_ghost) > 0:
            if gMDist <= 5:
                self.survival_mode = True
                self.safe_space = self.start
        elif myState.is_pacman and food_carrying > 2 and len(chase_ghost) > 0:
            if food_carrying > (self.totalFood / 4) or gDist <= 5:
                self.survival_mode = True
        elif food_left <= 2 or (food_carrying > 0 and self.move_count > 270):
            self.survival_mode = True
            self.safe_space = self.start

        # If in survival mode, check if the current survival point is safe
        if self.survival_mode and len(chase_ghost) > 0:
            ghost_positions = [a.get_position() for a in chase_ghost]
            home_distance = self.get_maze_distance(my_pos, self.safe_space)
            ghost_to_home_distance = min([self.get_maze_distance(gp, self.safe_space) for gp in ghost_positions])

            # Check if the survival point is valid; if not, find a safer point
            if ghost_to_home_distance < home_distance:
                for hp in self.escapeHatch:
                    home_distance = self.get_maze_distance(my_pos, hp)
                    ghost_to_home_distance = min([self.get_maze_distance(gp, hp) for gp in ghost_positions])
                    if home_distance < ghost_to_home_distance:
                        self.safe_space = hp
                        break

    def power_check(self, game_state):
        """
        Checks if the agent should enter power mode, which happens when both 
        opponent ghosts are scared and the scared time is sufficiently long.
        Args:
            game_state: The current state of the game.
        Sets:
            self.power_mode: Boolean indicating if the agent is in power mode.
            self.survival_mode: Boolean indicating if the agent is in survival mode.
            self.safe_space: The target point for survival mode.
        """
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]

        # Find the minimum scared time among the enemy ghosts
        min_ghost_scare_time = min(enemy.scared_timer for enemy in enemies)

        # Enter power mode if both ghosts are sufficiently scared
        if min_ghost_scare_time >= 15:
            self.power_mode = True
            self.survival_mode = False  # Exit survival mode when in power mode
            self.safe_space = self.start
        else:
            self.power_mode = False

    def generate_successors(self, game_state):
        """
        Generates all successor states from the current game state based on legal actions.
        Excludes actions that lead to a 'STOP' or to an empty path.

        Args:
            game_state: The current state of the game.

        Returns:
            A list of successor states.
        """
        # Get all legal actions for the current state, excluding 'STOP'
        actions = game_state.get_legal_actions(self.index)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)

        # Filter out actions that lead to an empty path
        valid_actions = [action for action in actions if not self.check_empty_path(game_state, action, 8)]

        # Generate and return the successor states for all valid actions
        return [self.get_successor(game_state, action) for action in valid_actions]
   
    def MCT(self, game_state, iter_times, simulate_depth):
        """
        Constructs and traverses a Monte Carlo tree based on the current game state.
        Args:
            game_state: The current state of the game.
            iter_times: Number of iterations for MCTS.
            simulate_depth: Depth for each simulation in the tree.
        Returns:
            The root node of the Monte Carlo tree after traversal.
        """
        startTime = time.time()
        root = MCTNode(game_state, None, 0.0, 0)  # Initialize root node with the current game state

        # Expand the root node by adding child nodes
        successors = self.generate_successors(root.game_state)
        for suc in successors:
            child = MCTNode(suc, root, 0.0, 0)
            root.addChild(child)

        for i in range(iter_times):
            curNode = root
            if time.time() - startTime > 0.95:  # Break if nearing time limit (e.g., 0.95 seconds)
                break

            # Selection
            while curNode.children is not None:

                uct_values = [self.uct(child, child.parent) for child in curNode.children]
                
                # Check if uct_values is empty
                if not uct_values:
                    # Handle the empty sequence case
                    # For example, you can break the loop, expand the current node, or choose a default action
                    break
            
          
                max_uct_value = max(uct_values)
                curNode = [c for c, v in zip(curNode.children, uct_values) if v == max_uct_value][0]
                children = curNode.children
            # Expansion: Add new child nodes to the current node
            if curNode.visits != 0:
                successors = self.generate_successors(curNode.game_state)
                for suc in successors:
                    child = MCTNode(suc, curNode, 0.0, 0)
                    curNode.addChild(child)
                curNode = curNode.children[0]  # Select a child node for simulation

            # Simulation: Simulate the game from the current node and calculate reward
            reward = self.simulate(curNode.game_state, simulate_depth, 0.0)

            # Backpropagation: Update the nodes with the simulation result
            while curNode.parent is not None:
                curNode.reward += reward
                curNode.visits += 1
                curNode = curNode.parent
            # Update the root node as well
            root.reward += reward
            root.visits += 1

        return root

    def simulate(self, game_state, level, reward):
        """
        Performs a simulation in MCTS by taking random actions until a specified level is reached.
        Args:
            game_state: The current state of the game.
            level: The depth level for the simulation.
            reward: The current accumulated reward.
        Returns:
            The accumulated reward after the simulation.
        """
        if level == 0:
            # Base case: If recursion level is 0, return the accumulated reward
            return reward

        # Choose a random legal action (excluding STOP)
        legalActions = game_state.get_legal_actions(self.index)
        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)
        next_action = random.choice(legalActions)

        # Generate the successor state and position after taking the action
        successor = self.get_successor(game_state, next_action)
        successorPos = successor.get_agent_position(self.index)

        # Update the reward based on the state and action
        currentReward = reward + self.evaluate(game_state, next_action, self.survival_mode)

        # Update the closest ghost position if the agent is not Pacman
        if not successor.get_agent_state(self.index).is_pacman:
            ghosts = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
            nonPacmanGhosts = [ghost for ghost in ghosts if ghost.get_position() is not None and not ghost.is_pacman]
            if nonPacmanGhosts:
                closestGhostDist = min(self.get_maze_distance(successorPos, ghost.get_position()) for ghost in nonPacmanGhosts)
                closestGhost = min(nonPacmanGhosts, key=lambda ghost: self.get_maze_distance(successorPos, ghost.get_position()))
                self.ghost_pos = closestGhost.get_position() if closestGhostDist < 99999.9 else self.ghost_pos

        # Recur with the updated level and reward
        return self.simulate(successor, level - 1, currentReward)

    def uct(self, node, parentNode):
        """
        Calculates the Upper Confidence Bound for Trees (UCT) value for a node in MCTS.

        Args:
            node: The node for which UCT value is being calculated.
            parentNode: The parent of the node.

        Returns:
            The UCT value for the node.

        Note:
            Cp is the exploration parameter. A higher Cp values encourage more exploration,
            while lower values emphasize exploitation of known good paths.
        """
        # Exploration parameter
        Cp = 0.75

        # Number of times the node has been visited
        visits = node.visits

        if visits > 0:
            # UCB1 formula applied for nodes that have been visited
            ucb = 2 * Cp * math.sqrt(2 * math.log(parentNode.visits) / visits)
        else:
            # Assign a high UCT value to encourage exploration of unvisited nodes
            ucb = float('inf')
        
        # Combine the node's reward with the UCB value
        return node.reward + ucb

    def get_features(self, game_state, action):
        """
        Extracts features from a given game state and action.
        Args:
            game_state: The current state of the game.
            action: The action to be evaluated.
        Returns:
            A util.Counter object containing the extracted features.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        myState = successor.get_agent_state(self.index)
        my_pos = myState.get_position()
        # print("MY potential posiiton: ", my_pos)
        
        # Food and capsules
        food_grid = self.get_food(successor)
        
        food_positions = []
        for x in range(game_state.data.layout.width):  # Iterate over each row
            for y in range(game_state.data.layout.height):  # Iterate over each column in the row
                if food_grid[x][y]:  # Check if there's food at position (x, y)
                    food_positions.append((x, y))  # Append the position to the list
        
        capsuleList = self.get_capsules(successor)
        features['successor_score'] = -len(food_positions)  # Negative of food count
        features['eatenCap'] = -len(capsuleList)    # Negative of capsule count

        # Avoid reverse action
        # print("Action", action)
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        features['reverse'] = 1 if action == rev else 0

        # Distance to the nearest food
        if food_positions:
            min_distance = min(self.get_maze_distance(my_pos, food) for food in food_positions)
            features['distance_to_food'] = min_distance

        # Distance to observable enemies
        features['distance_to_op'] = self.get_distance_to_observable_enemies(successor, my_pos)

        # Distance to the best entry point
        # print("Am i pacman: ", self.index, game_state.get_agent_state(self.index).is_pacman)
        if not game_state.get_agent_state(self.index).is_pacman:
            features['distance_to_entry'] = self.get_distance_to_best_entry(my_pos)

        # Distance to the closest home point if in survival mode
        if self.survival_mode:
            minHomeDist = min(self.get_maze_distance(my_pos, hp) for hp in self.escapeHatch)
            features['distance_to_home'] = minHomeDist

        # Feature indicating if Pacman arrived home safely
        features['arrived_home'] = self.get_score(successor)

        # Feature indicating if the agent is 'dead' (at initial position)
        features['dead'] = 1 if my_pos == self.initialPosition else 0

        # Feature indicating if the agent is at a dead end
        features['dead_end'] = 1 if self.survival_mode and len(successor.get_legal_actions(self.index)) <= 1 else 0

        return features

    def get_distance_to_observable_enemies(self, successor, my_pos):
        """
        Computes the distance to observable enemy agents.
        Args:
            successor: The successor game state.
            my_pos: The current position of the agent.
        Returns:
            The distance to the closest observable enemy, or -100 if very close.
        """
        distances = []
        for op in self.get_opponents(successor):
            # print("opponent agent index:" , op)
            opState = successor.get_agent_state(op)
            # print("Opponent state: ", successor.get_agent_state(op))
            # print("Is pacman? ", opState.is_pacman)
            # print("Position: ", opState.get_position())
            if not opState.is_pacman and opState.get_position() is not None:
                # print("Inside for loop")
                distance = self.get_maze_distance(my_pos, opState.get_position())
                distances.append(-100 if distance == 1 else distance) 
            # print("------------")
        return min(distances) if distances else 0

    def get_distance_to_best_entry(self, my_pos):
        """
        Computes the distance to the best entry point (to the offensive side) based on ghost positions.
        Args:
            my_pos: The current position of the agent.
        Returns:
            The distance to the best entry point.
        """
        best_entry = max(self.entryPoints, key=lambda ep: self.get_maze_distance(self.ghost_pos, ep))
        # print("BEST ENTRY POINT =", best_entry)
        return self.get_maze_distance(my_pos, best_entry)

    def get_weights(self, game_state, action):
        """
        Returns weights for each feature used in evaluating actions.
        Args:
            game_state: The current state of the game.
            action: The action to be evaluated.
        Returns:
            A dictionary of weights for each feature.
        """
        if self.power_mode:
            return {'successor_score': 150, 'distance_to_food': -10, 'reverse': -3, 'distance_to_op': -15,'dead': -200, 'dead_end': 0, 'eatenCap': 0}
        return {'successor_score': 150, 'distance_to_food': -5, 'reverse': -3, 'distance_to_entry': -10,'dead': -200, 'dead_end': -100, 'eatenCap': 200}

    def evaluate(self, game_state, action, survival_mode):
        """
        Evaluates the desirability of a given action based on the current game state.
        This is done by computing a linear combination of various features and their weights.
        
        Args:
            game_state: The current state of the game.
            action: The action to be evaluated.
            survival_mode: Boolean indicating if the agent is in survival mode.

        Returns:
            The value of the action, calculated as a dot product of features and weights.
        """
        # Extract features for the given action
        features = self.get_features(game_state, action)

        # Get the corresponding weights for the features
        weights = self.get_weights(game_state, action)

        # Print the weights used in evaluation (for debugging and analysis)
        # print("position: ", self.get_successor(game_state, action).get_agent_state(self.index).get_position())
        # print("Features: ", self.get_features(game_state, action))
        # print("weights`; ", self.get_weights(game_state, action))
        # print("\n--------------------------------------------\n")

        # Return the dot product of features and weights
        # This value represents the desirability of the action
        return features * weights

   


class DefensiveReflexAgent(ReflexCaptureAgent):
    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        # self.init_food = self.get_food_you_are_defending(game_state).asList()
        self.init_food = []
        food_grid = self.get_food_you_are_defending(game_state)  # Remaining food in the game
        # Iterate over the grid to count food items
        for x in range(food_grid.width):
            for y in range(food_grid.height):
                if food_grid[x][y]:  # Assuming this checks for food at the (x, y) position
                    self.init_food.append((x,y))

        self.chase_dest = []
        height = (game_state.data.layout.height - 2) / 2
        i = 0

        # Calculate the central point of the map, and set it as the defender initial position
        if self.red:
            for central in range(int((game_state.data.layout.width - 2) / 2), 0, -1):
                central = int(central)
                height = int(height)

                if not game_state.has_wall(central, height):
                    self.detectDest = [(central, height)]
                    i = i + 1
                    if i == 2:
                        break

        else:
            for central in range(
                (game_state.data.layout.width - 2) / 2 + 1,
                game_state.data.layout.width,
                1,
            ):
                if not game_state.has_wall(central, height):
                    central = int(central)
                    height = int(height)

                    self.detectDest = [(central, height)]
                    i = i + 1
                    if i == 2:
                        break

    def choose_action(self, game_state):
        """
        Chooses an action for the agent based on the current game state. This method evaluates 
        possible actions and decides the best course of action based on various scenarios such as 
        the presence of invaders, the need to defend or attack, and the state of the agent (scared or normal).
        """

        # Retrieve all legal actions for the agent in the current game state
        actions = game_state.get_legal_actions(self.index)

        # Evaluate the game state for each action to find their respective values
        values = [self.evaluate(game_state, a) for a in actions]

        # Find the maximum value among all evaluated actions
        max_value = max(values)
        # Select all actions that have this maximum value
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # Get current enemy agents
        curr_enemies = [
            game_state.get_agent_state(i) for i in self.get_opponents(game_state)
        ]
        # Identify if there are any invaders (enemy Pacman)
        curr_invader = [
            enemy
            for enemy in curr_enemies
            if enemy.is_pacman and enemy.get_position() != None
        ]

        # Get the current food distribution in your territory
        food_list = []
        food_grid = self.get_food_you_are_defending(game_state)
        # Construct a list of food positions
        for x in range(food_grid.width):
            for y in range(food_grid.height):
                if food_grid[x][y]:  # Check if there is food at (x, y)
                    food_list.append((x, y))

        # Current agent's position
        loc = game_state.get_agent_state(self.index).get_position()

        # Check for eaten food and update chase destination accordingly
        if len(self.init_food) - len(food_list) > 0:
            eaten_food = list(set(self.init_food).difference(set(food_list)))
            self.init_food = food_list
            self.chase_dest = eaten_food
        agent_state = game_state.get_agent_state(self.index)

        # Attack mode: when the agent is not scared or the scared timer is high
        if agent_state.scared_timer > 10:
            # Evaluate the attack potential of each action
            attackValues = [self.evaluate_attack(game_state, a) for a in actions]
            maxAttackValue = max(attackValues)
            bestAttackActions = [
                a for a, v in zip(actions, attackValues) if v == maxAttackValue
            ]
            return random.choice(bestAttackActions)

        # Defense mode: when the agent is scared or the scared timer is low
        elif agent_state.scared_timer <= 10:
            if len(curr_invader) > 0:
                self.chase_dest = []
            elif len(self.chase_dest) > 0:
                # Use A* algorithm to find the path to the chase destination
                self.chaseFood = self.AStar(game_state, self.chase_dest[0])
                if len(self.chaseFood) > 0:
                    return self.chaseFood[0]
                if loc == self.chase_dest[0]:
                    self.chase_dest = []

            elif len(self.detectDest) > 0:
                # Use A* algorithm to find the path to the detect destination
                self.searchActions = self.AStar(game_state, self.detectDest[0])
                if len(self.searchActions) > 0:
                    return self.searchActions[0]

        # If no specific action is required, choose randomly among the best actions
        return random.choice(best_actions)


    def AStar(self, game_state, destination):
        """
        Performs A* search algorithm to find a path from the agent's current location to a specified destination.

        Args:
        game_state: The current state of the game.
        destination: The target location to reach.

        Returns:
        A list of actions that leads from the agent's current position to the destination.
        """
        from util import PriorityQueue

        # Initialize data structures for A* search
        visited, movements, costs, start_cost = [], {}, {}, 0
        start_agent_state = game_state.get_agent_state(self.index)  # Get the starting state of the agent
        startLoc = start_agent_state.get_position()  # Get the starting location of the agent
        movements[startLoc] = []  # Initialize the movement path for the starting location
        costs[startLoc] = 0  # Set the initial cost to zero
        visited.append(startLoc)  # Mark the start location as visited

        # Initialize priority queue for the A* search
        priorityQueue = PriorityQueue()
        priorityQueue.push(game_state, start_cost)

        # Loop until the priority queue is empty
        while not priorityQueue.isEmpty():
            currGame_state = priorityQueue.pop()  # Get the game state with the lowest cost
            currLoc = currGame_state.get_agent_state(self.index).get_position()  # Current location of the agent

            # Check if the current location is the destination
            if currLoc == destination:
                return movements[currLoc]  # Return the path to the destination

            # Explore the successor states
            actions = currGame_state.get_legal_actions(self.index)
            for action in actions:
                succGame_state = self.get_successor(currGame_state, action)  # Get successor state for the action
                succLoc = succGame_state.get_agent_state(self.index).get_position()  # Get the location in the successor state
                newCost = 1  # Cost of moving to the successor location, assuming a cost of 1 per move
                next_cost = costs[currLoc] + newCost  # Update the total cost to reach the successor location

                # Check if the successor location is either unvisited or can be reached with a lower cost
                if succLoc not in visited or next_cost < costs[succLoc]:
                    visited.append(succLoc)  # Mark the successor location as visited
                    movements[succLoc] = []  # Initialize the movement path for the successor location
                    pre = movements[currLoc][:]  # Copy the current path
                    pre.append(action)  # Add the current action to the path
                    movements[succLoc].extend(pre)  # Update the path for the successor location
                    costs[succLoc] = next_cost  # Update the cost for the successor location
                    heurisitic = self.Heuristic(succLoc, destination)  # Calculate the heuristic from the successor location to the destination
                    priority = next_cost + heurisitic  # Calculate the priority for the priority queue
                    priorityQueue.push(succGame_state, priority)  # Push the successor state and its priority to the priority queue

    def Heuristic(self, loc, destination):
        """
        Use Manhattan distance to calculate the heuristic distance.
        """
        from util import manhattanDistance

        dist = manhattanDistance(loc, destination)
        return dist

    def isGoal(self, game_state):
        agent_state = game_state.get_agent_state(self.index)
        if agent_state.is_pacman:
            return True
        else:
            return False

    def get_features(self, game_state, action):

        features = util.Counter()
        successor_game_state = self.get_successor(game_state, action)
        successor_agent_state = successor_game_state.get_agent_state(self.index)
        myPos = successor_agent_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features["onDefense"] = 1
        if successor_agent_state.is_pacman:
            features["onDefense"] = 0
        if action == Directions.STOP:
            features["stop"] = 1

        # Computes distance to invaders we can see
        enemies = [
            successor_game_state.get_agent_state(i)
            for i in self.get_opponents(successor_game_state)
        ]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() != None]
        features["numInvaders"] = len(invaders)
        if len(invaders) > 0:
            min_dist = min(
                [self.get_maze_distance(myPos, a.get_position()) for a in invaders]
            )
            features["invaderDistance"] = min_dist
            if successor_agent_state.scared_timer > 0 and min_dist == 0:
                features["scared"] = 1
                features["stop"] = 0

        return features

    def get_weights(self, game_state, action):
        return {
            "numInvaders": -1000,
            "onDefense": 1000,
            "invaderDistance": -10,
            "stop": -100,
            "scared": -1000,
        }

    def evaluate_attack(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_attack_features(game_state, action)
        weights = self.get_attack_weights(game_state, action)
        return features * weights

    def get_attack_features(self, game_state, action):
        features = util.Counter()
        if action == Directions.STOP:
            features["stop"] = 1

        succGame_state = self.get_successor(game_state, action)
        actions = succGame_state.get_legal_actions(self.index)
        succ_agent_state = succGame_state.get_agent_state(self.index)
        succPos = succ_agent_state.get_position()
        enemies = [
            succGame_state.get_agent_state(i) for i in self.get_opponents(succGame_state)
        ]
        ghosts = [
            enemy
            for enemy in enemies
            if not enemy.is_pacman and enemy.get_position() != None
        ]
        invaders = [
            enemy
            for enemy in enemies
            if enemy.is_pacman and enemy.get_position() != None
        ]
        capsules = self.get_capsules(succGame_state)

        # run away from ghost
        features["distanceToGhost"] = 99999
        if len(ghosts) > 0:
            min_distToGhost = min(
                [
                    self.get_maze_distance(succPos, ghost.get_position())
                    for ghost in ghosts
                ]
            )
            nearestGhost = [
                ghost
                for ghost in ghosts
                if self.get_maze_distance(succPos, ghost.get_position())
                == min_distToGhost
            ]
            if nearestGhost[0].scared_timer > 0:
                features["distanceToGhost"] = 99999
            elif succ_agent_state.is_pacman:
                features["distanceToGhost"] = min_distToGhost

        # eat capsules
        features["remainCap"] = len(capsules)
        if len(capsules) > 0:
            features["distanceToCap"] = min(
                [self.get_maze_distance(succPos, cap) for cap in capsules]
            )

        # eat food
        food_list = self.get_food(succGame_state).asList()
        features["remainFood"] = len(food_list)

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            features["distanceToFood"] = min(
                [self.get_maze_distance(succPos, food) for food in food_list]
            )

        return features

    def get_attack_weights(self, game_state, action):
        return {
            "remainFood": -100,
            "distanceToFood": -1,
            "remainCap": -100,
            "distanceToCap": -1,
            "distanceToGhost": 1000,
            "stop": -3000,
        }

class MCTNode:
    def __init__(self, game_state, parent, reward, action=None):
        self.game_state = game_state        # The game state at this node
        self.parent = parent      # The node's parent
        self.action = action      # The action that led to this state
        self.children = []        # List to store the node's children
        self.visits = 0           # Number of times the node has been visited
        self.reward = reward    # Cumulative value of the node

    def addChild(self, child_node):
        """
        Adds a child node to this node
        """
        self.children.append(child_node)

    # def update(self, value):
    #     """
    #     Updates the node's statistics based on the simulation result
    #     """
    #     self.visits += 1
    #     self.value += value




