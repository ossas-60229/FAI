# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random
import util
import sys

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        def the_dist(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        now_legal_actions = successorGameState.getLegalActions()
        newGhostPos = successorGameState.getGhostPositions()
        GhostNum = len(newGhostPos)
        FoodNum = len(newFood.asList())
        ret_score = successorGameState.getScore()
        min_to_g = sys.maxsize
        ind_g = 0
        direct_g = 0
        for i in range(GhostNum):
            to_g = the_dist(newPos, newGhostPos[i])
            if to_g < min_to_g:
                min_to_g = to_g
                ind_g = i
        min_to_f = sys.maxsize
        for i in range(FoodNum):
            to_f = the_dist(newPos, newFood.asList()[i])
            food_pos = newFood.asList()[i]
            offset = 100
            times = 0
            outa = 0
            # to NORTH
            if food_pos[1] > newPos[1]:
                outa += 1
                if 'North' not in now_legal_actions:
                    times += 1
            # to SOUTH
            if food_pos[1] < newPos[1]:
                outa += 1
                if 'South' not in now_legal_actions:
                    times += 1
            # to EAST
            if food_pos[0] > newPos[0]:
                outa += 1
                if 'East' not in now_legal_actions:
                    times += 1
            # to WEST
            if food_pos[0] < newPos[0]:
                outa += 1
                if 'West' not in now_legal_actions:
                    times += 1
            if outa == times:
                to_f += offset*times
            if to_f < min_to_f:
                min_to_f = to_f
        if action == 'Stop':
            ret_score -= 50
        KEIKAI = 2
        if GhostNum == 0:
            min_to_g = KEIKAI + 10
        if min_to_g <= KEIKAI and newScaredTimes[ind_g] < KEIKAI*2:
            ret_score -= 250/(min_to_g + 1)
        else:
            ret_score += 200 / (min_to_f*FoodNum*10 + 1)
        return ret_score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.Minimax_eval(gameState, 0, 0, -sys.maxsize)[1]
    # evaluation function (MinMax ver.)

    def Minimax_eval(self, gameState: GameState, index: int, depth: int, maxmin: float):
        # the leaf state
        legal_moves = gameState.getLegalActions(index)
        ghost_num = gameState.getNumAgents() - 1
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), ""
        else:
            # This is PACMAN (max agent)

            if index == 0:
                max_val = -sys.maxsize
                max_move = ""
                for lmove in legal_moves:
                    successor = gameState.generateSuccessor(index, lmove)
                    new_index = index + 1
                    new_depth = depth
                    tmp_val = self.Minimax_eval(
                        successor, new_index, new_depth, max_val)[0]
                    if tmp_val > max_val:
                        max_val = tmp_val
                        max_move = lmove
                return (max_val, max_move)
            # This is GHOST (min agent)
            elif index > 0:
                min_val = sys.maxsize
                min_move = ""
                for lmove in legal_moves:
                    successor = gameState.generateSuccessor(index, lmove)
                    new_index = index + 1
                    new_depth = depth
                    if new_index > ghost_num:
                        new_index = 0
                        new_depth += 1
                    tmp_val = self.Minimax_eval(
                        successor, new_index, new_depth, maxmin)[0]
                    # if tmp_val < maxmin:
                    # maxmin = tmp_val
                    # min_move = lmove
                    # break
                    if tmp_val < min_val:
                        min_val = tmp_val
                        min_move = lmove
                return (min_val, min_move)
        return (0, "")


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.alphabeta_eval(gameState, 0, 0, -sys.maxsize, sys.maxsize)[1]
    # evaluation function (AlphaBeta ver.)

    def alphabeta_eval(self, gameState: GameState, index: int, depth: int, alpha: float, beta: float):
        # the leaf state
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), ""
        legal_moves = gameState.getLegalActions(index)
        ghost_num = gameState.getNumAgents() - 1
        # This is PACMAN (max agent)

        if index == 0:
            max_val = -sys.maxsize
            tmp_alpha = alpha
            max_move = ""
            for lmove in legal_moves:
                successor = gameState.generateSuccessor(index, lmove)
                new_index = index + 1
                new_depth = depth
                tmp_alpha = max(tmp_alpha, max_val)
                tmp_val = self.alphabeta_eval(
                    successor, new_index, new_depth, tmp_alpha, beta)[0]
                if tmp_val > max_val:
                    max_val = tmp_val
                    max_move = lmove
                if max_val > beta:
                    return (max_val, max_move)
            return (max_val, max_move)
        # This is GHOST (min agent)
        elif index > 0:
            min_val = sys.maxsize
            min_move = ""
            tmpbeta = beta
            for lmove in legal_moves:
                successor = gameState.generateSuccessor(index, lmove)
                new_index = index + 1
                new_depth = depth
                if new_index > ghost_num:
                    new_index = 0
                    new_depth += 1
                # if tmp_val < maxmin:
                # maxmin = tmp_val
                # min_move = lmove
                # break
                tmpbeta = min(tmpbeta, min_val)
                tmp_val = self.alphabeta_eval(
                    successor, new_index, new_depth, alpha, tmpbeta)[0]
                if tmp_val < min_val:
                    min_val = tmp_val
                    min_move = lmove
                if min_val < alpha:
                    return (min_val, min_move)
            return (min_val, min_move)
        return (0, "")


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.Expectimax_eval(gameState, 0, 0, 0)[1]

    def Expectimax_eval(self, gameState: GameState, index: int, depth: int, maxmin: float):
        # the leaf state
        legal_moves = gameState.getLegalActions(index)
        ghost_num = gameState.getNumAgents() - 1
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), ""
        else:
            # This is PACMAN (max agent)
            if index == 0:
                max_val = -sys.maxsize
                max_move = ""
                for lmove in legal_moves:
                    successor = gameState.generateSuccessor(index, lmove)
                    new_index = index + 1
                    new_depth = depth
                    tmp_val = self.Expectimax_eval(
                        successor, new_index, new_depth, max_val)[0]
                    if tmp_val > max_val:
                        max_val = tmp_val
                        max_move = lmove
                return (max_val, max_move)
            # This is GHOST (expect agent)
            elif index > 0:
                min_val = sys.maxsize
                min_move = ""
                exp_val = 0
                for lmove in legal_moves:
                    successor = gameState.generateSuccessor(index, lmove)
                    new_index = index + 1
                    new_depth = depth
                    if new_index > ghost_num:
                        new_index = 0
                        new_depth += 1
                    tmp_val = self.Expectimax_eval(
                        successor, new_index, new_depth, maxmin)[0]
                    exp_val += tmp_val
                min_val = float(exp_val) / len(legal_moves)
                min_move = ""
                return (min_val, min_move)
        return (0, "")


def betterEvaluationFunction(currentGameState: GameState):
    if currentGameState.isWin():
        return scoreEvaluationFunction(currentGameState) + 10000
    if currentGameState.isLose():
        return -10000

    def the_dist(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    desciption: 
        1.calculate the lowest distance between pacman and ghost
        2.if it is dangerous, applay alphabeta pruning
        3.if it is safe, apply greedy method(eat food first)
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    Gnum = currentGameState.getNumAgents() - 1

    def alphabeta_eval(gameState: GameState, index: int, depth: int, alpha: float, beta: float):
        # the leaf state
        if depth == 2 or gameState.isWin() or gameState.isLose():
            return scoreEvaluationFunction(gameState), ""
        legal_moves = gameState.getLegalActions(index)
        ghost_num = gameState.getNumAgents() - 1
        # This is PACMAN (max agent)

        if index == 0:
            max_val = -sys.maxsize
            tmp_alpha = alpha
            max_move = ""
            for lmove in legal_moves:
                successor = gameState.generateSuccessor(index, lmove)
                new_index = index + 1
                new_depth = depth
                tmp_alpha = max(tmp_alpha, max_val)
                tmp_val = alphabeta_eval(
                    successor, new_index, new_depth, tmp_alpha, beta)[0]
                if lmove == "stop":
                    tmp_val = -sys.maxsize
                if tmp_val > max_val:
                    max_val = tmp_val
                    max_move = lmove
                if max_val > beta:
                    return (max_val, max_move)
            return (max_val, max_move)
        # This is GHOST (min agent)
        elif index > 0:
            min_val = sys.maxsize
            min_move = ""
            tmpbeta = beta
            for lmove in legal_moves:
                successor = gameState.generateSuccessor(index, lmove)
                new_index = index + 1
                new_depth = depth
                if new_index > ghost_num:
                    new_index = 0
                    new_depth += 1
                # if tmp_val < maxmin:
                # maxmin = tmp_val
                # min_move = lmove
                # break
                tmpbeta = min(tmpbeta, min_val)
                tmp_val = alphabeta_eval(
                    successor, new_index, new_depth, alpha, tmpbeta)[0]
                if tmp_val < min_val:
                    min_val = tmp_val
                    min_move = lmove
                if min_val < alpha:
                    return (min_val, min_move)
            return (min_val, min_move)
        return (0, "")
    ghost_poss = currentGameState.getGhostPositions()
    pacman_poss = currentGameState.getPacmanPosition()
    ghost_dist = [(the_dist(ghost_poss[i], pacman_poss), i + 1)
                  for i in range(Gnum)]
    ghost_dist.sort()
    food_pos = currentGameState.getFood().asList()
    min_f_dist = sys.maxsize
    min_food = food_pos[0]
    legal_moves = currentGameState.getLegalActions(0)
    for fpos in food_pos:
        dist = the_dist(fpos, pacman_poss)
        offset = 20
        time = 0
        ff = 0
        if min_food[0] > pacman_poss[0]:
            ff += 1
            if "East" not in legal_moves:
                time += 1
        if min_food[0] < pacman_poss[0]:
            ff += 1
            if "West" not in legal_moves:
                time += 1
        if min_food[1] < pacman_poss[1]:
            ff += 1
            if "South" not in legal_moves:
                time += 1
        if min_food[1] > pacman_poss[1]:
            ff += 1
            if "North" not in legal_moves:
                time += 1
        if ff == time:
            dist += offset*time
        if dist < min_f_dist:
            min_f_dist = dist
            min_food = fpos
    if ghost_dist[0][0] <= 3:
        ffss = alphabeta_eval(
            currentGameState, 0, 0, -sys.maxsize, sys.maxsize)
        if (ffss[1] == "Stop"):
            return -sys.maxsize
        else:
            return ffss[0]
    else:
        min_max_val = scoreEvaluationFunction(
            currentGameState) + ghost_dist[0][0]/(min_f_dist*10 + 1)
    return min_max_val


# Abbreviation
better = betterEvaluationFunction
