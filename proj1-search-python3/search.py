# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    visited = set()
    stack = util.Stack()
    # need a tuple for state (x,y) and actions which is Directions like "West","South"
    stack.push((problem.getStartState(), []))
    while not stack.isEmpty():
        popped = stack.pop()
        state = popped[0]
        actions = popped[1]

        if state in visited:
            continue

        visited.add(state)

        print("state:",state,"action:",actions)
        if problem.isGoalState(state):
            return actions

        for nextState, direction, cost in problem.getSuccessors(state):
            if nextState not in visited:
                """
                #First I try to do actions.append(direction) in the stack.push, but all I have in actions are None.
                Then I did a test in python3,
                >>> a=[]
                >>> b=a.append("32")
                >>> b == None
                True
                Seems like you cannot make two actions at the same time. you cannot call function append and assignment
                at the same time. It's gonna cause some trouble. I remember in cs61a, during for loop, 
                you can't delete elements, like you can't do 
                a = [1,2,3]
                for i in a:
                    a.pop()
                you will end up getting a=[1]
                instead if you do 
                for i in list(a):
                    a.pop()
                    
                it will work
                """

                stack.push((nextState,actions + [direction]))

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    visited = set()
    queue = util.Queue()

    queue.push((problem.getStartState(),[]))
    while not queue.isEmpty():
        popped = queue.pop()
        state = popped[0]
        actions = popped[1]

        if state in visited:
            continue

        visited.add(state)
        if problem.isGoalState(state):
            return actions
        for nextState, direction, cost in problem.getSuccessors(state):
            if nextState not in visited:
                queue.push((nextState,actions+[direction]))

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    p_queue = util.PriorityQueue()
    visited = set()
    p_queue.push((problem.getStartState(), []), 0)
    while not p_queue.isEmpty():
        popped = p_queue.pop()
        state = popped[0]
        actions = popped[1]

        if state in visited:
            continue

        visited.add(state)
        if problem.isGoalState(state):
            return actions
        for nextState, direction, nextCost in problem.getSuccessors(state):
            if nextState not in visited:
                p_queue.push((nextState, actions + [direction]), nextCost+problem.getCostOfActions(actions))
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    p_queue = util.PriorityQueue()
    visited = set()
    p_queue.push((problem.getStartState(), []), 0)
    while not p_queue.isEmpty():
        popped = p_queue.pop()
        state = popped[0]
        actions = popped[1]

        if state in visited:
            continue

        visited.add(state)

        if problem.isGoalState(state):
            return actions

        for nextState, direction, nextCost in problem.getSuccessors(state):
            if nextState not in visited:
                p_queue.push(
                    (nextState, actions+[direction]),
                    nextCost+problem.getCostOfActions(actions) + heuristic(nextState, problem = problem))
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
