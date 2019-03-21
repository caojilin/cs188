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
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        if successorGameState.isWin():
            return 10 * 100
        for ghostState in newGhostStates:
            if manhattanDistance(ghostState.getPosition(), newPos) <= 1:
                return -10 * 100

        def closestFood(foodArray):
            return min(manhattanDistance(food, newPos) for food in foodArray)

        foodPos = list(newFood.asList())
        "*** YOUR CODE HERE ***"

        return -len(foodPos) + 0.5 / (closestFood(foodPos) + 1)


def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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

        def miniMax_search(state, depth, agent):
            if agent == state.getNumAgents():
                if depth == self.depth:
                    return self.evaluationFunction(state)
                else:
                    return miniMax_search(state, depth + 1, 0)
            else:
                actions = state.getLegalActions(agent)

                if len(actions) == 0:
                    return self.evaluationFunction(state)

                nextStates = (miniMax_search(state.generateSuccessor(agent, action), depth, agent + 1)
                              for action in actions)

                if agent == 0:
                    return max(nextStates)
                else:
                    return min(nextStates)

        possibleMoves = gameState.getLegalActions(0)
        results = (miniMax_search(gameState.generateSuccessor(0, move), 1, 1) for move in possibleMoves)
        d = dict(zip(possibleMoves, results))
        return max(d, key=d.get)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def value(gameState, depth, agent, alpha, beta):
            if depth == self.depth or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            elif agent == 0:
                return maxValue(gameState, depth, alpha, beta)
            else:
                return minValue(gameState, depth, agent, alpha, beta)

        def maxValue(gameState, depth, alpha, beta):
            v = -10 * 100
            for action in gameState.getLegalActions(0):
                v = max(v, value(gameState.generateSuccessor(0, action), depth, 1, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def minValue(gameState, depth, agent, alpha, beta):
            v = 10 * 100
            for action in gameState.getLegalActions(agent):
                if agent == gameState.getNumAgents() - 1:
                    v = min(v, value(gameState.generateSuccessor(agent, action), depth + 1, 0, alpha, beta))
                else:
                    v = min(v, value(gameState.generateSuccessor(agent, action), depth, agent + 1, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        rootValue = -10 * 100
        alpha = -10 * 100
        beta = 10 * 100
        maxAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = value(nextState, 0, 1, alpha, beta)
            if nextValue > rootValue:
                rootValue = nextValue
                maxAction = action
            alpha = max(rootValue, alpha)
        return maxAction

        # return max(gameState.getLegalActions(0), key=lambda x: value(gameState.generateSuccessor(0, x),
        #                                                              0, 1, alpha, beta))


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        def average(*values):
            return sum(values) / len(values)

        def value(gameState, depth, agent):
            if depth == self.depth or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            elif agent == 0:
                return maxValue(gameState, depth)
            else:
                return avgValue(gameState, depth, agent)

        def maxValue(gameState, depth):
            v = -10*100
            for action in gameState.getLegalActions(0):
                v = max(v, value(gameState.generateSuccessor(0, action), depth, 1))
            return v

        def avgValue(gameState, depth, agent):
            v = 0
            for action in gameState.getLegalActions(agent):
                if agent == gameState.getNumAgents() - 1:
                    v = v + average(value(gameState.generateSuccessor(agent, action), depth + 1, 0))
                else:
                    v = v + average(value(gameState.generateSuccessor(agent, action), depth, agent + 1))
            v = v/len(gameState.getLegalActions(agent))
            return v

        rootValue = -10*100
        maxAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = value(nextState, 0, 1)

            if nextValue > rootValue:
                rootValue = nextValue
                maxAction = action
        return maxAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    we need to consider important factors in a given gamestate
    What we can have is number of foods, ghost positions, food positoins, win or lose
    if ghosts are too close, we need to decrease the overall eval value. The far the ghosts are, the better
    The close the foods are, the better. current score is a good indicator as well.
    Different factors have different weight. For example we need to consider when we eat the power dots,
    the ghosts are edible

    following points from wiki can help
    Pac-Dot - 10 points.
    Power Pellet - 50 points.
    Vulnerable Ghosts:
    #1 in succession - 200 points.
    #2 in succession - 400 points.
    #3 in succession - 800 points.
    #4 in succession - 1600 points.
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    # every move decrease the score
    score = currentGameState.getScore()

    ghostValue = 0
    for ghostState in newGhostStates:
        distance = manhattanDistance(pacmanPos, ghostState.getPosition())
        if distance > 0:
            if ghostState.scaredTimer > 0:
                ghostValue += 200 / distance
            else:
                ghostValue -= 50 / distance
    score += ghostValue

    foodPos = [manhattanDistance(pacmanPos, food) for food in list(newFood.asList())]
    if len(foodPos) != 0:
        score += -len(foodPos) + 0.5 / (min(foodPos) + 1)
    return score


# Abbreviation
better = betterEvaluationFunction
