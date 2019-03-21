# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        "Counter is a special dictionary."
        for i in range(self.iterations):
            valueState = util.Counter()
            for state in self.mdp.getStates():
                stateAction = util.Counter()
                for action in self.mdp.getPossibleActions(state):
                    stateAction[action] = self.computeQValueFromValues(state, action)
                valueState[state] = stateAction[stateAction.argMax()]
                # This handles the case that no actions available. Then stateAction.argMax() returns None
                # and stateAction[None] returns 0, this is not a normal dictionary can do.
            for state in self.mdp.getStates():
                self.values[state] = valueState[state]

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transitionProbabilities = self.mdp.getTransitionStatesAndProbs(state, action)
        q_value = 0
        for nextState, prob in transitionProbabilities:
            # The number of nextState depend on the action. So the complexity is bit-O (S^2A) for value iteration
            q_value += prob*(self.mdp.getReward(state, action, nextState) + self.discount*self.values[nextState])

        return q_value
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        possibleActions = self.mdp.getPossibleActions(state)
        if len(possibleActions) == 0:
            return

        valueAction = util.Counter()
        for action in possibleActions:
            valueAction[action] = self.computeQValueFromValues(state, action)

        return valueAction.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        totalStates = self.mdp.getStates()
        if self.iterations < len(totalStates):
            totalStates = totalStates[:self.iterations]
        else:
            remainder = self.iterations % len(totalStates)
            mod = self.iterations // len(totalStates)
            toAdd = totalStates[:remainder]
            totalStates = totalStates * mod
            totalStates.extend(toAdd)
        for state in totalStates:
            stateAction = util.Counter()
            for action in self.mdp.getPossibleActions(state):
                stateAction[action] = self.computeQValueFromValues(state, action)
            self.values[state] = stateAction[stateAction.argMax()]



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # Compute predecessors of all states
        predecessors = util.Counter()
        for state in self.mdp.getStates():
            predecessors[state] = set()
        # Initialize an empty priority queue
        p_queue = util.PriorityQueue()
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                qValue = []
                possibleActions = self.mdp.getPossibleActions(state)
                for action in possibleActions:
                    transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                    for nextState, prob in transitions:
                        if prob > 0:
                            predecessors[nextState].add(state)
                    qValue.append(self.getQValue(state, action))
                diff = abs(self.values[state] - max(qValue))
                p_queue.push(state, -diff)
        for i in range(self.iterations):
            if p_queue.isEmpty():
                return
            s = p_queue.pop()
            # Update s's value (if it is not a terminal state) in self.values.
            if not self.mdp.isTerminal(s):
                stateAction = util.Counter()
                for action in self.mdp.getPossibleActions(s):
                    stateAction[action] = self.computeQValueFromValues(s, action)
                self.values[s] = stateAction[stateAction.argMax()]

            for p in predecessors[s]:
                possibleActions = self.mdp.getPossibleActions(p)
                if len(possibleActions) == 0:
                    highestQ = 0
                else:
                    highestQ = max(self.getQValue(p, action) for action in possibleActions)
                diff = abs(self.values[p] - highestQ)
                if diff > self.theta:
                    p_queue.update(p, -diff)
