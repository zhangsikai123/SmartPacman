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
        for i in range(self.iterations):
            tempValues = self.values.copy()
            for s in self.mdp.getStates():
                maxValue = None
                for a in self.mdp.getPossibleActions(s):
                    temp = 0
                    for (nextState, P) in self.mdp.getTransitionStatesAndProbs(s, a):
                        reward = self.mdp.getReward(s, a, nextState)
                        value = tempValues[nextState]
                        temp += P * (reward + self.discount * value)

                    maxValue = max(temp, maxValue)
                if self.mdp.isTerminal(s):
                    self.values[s] = 0
                else:
                    self.values[s] = maxValue
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
        return sum(
            prob * (self.mdp.getReward(state, action, newState) +
                    self.discount * self.getValue(newState))
            for newState, prob in self.mdp.getTransitionStatesAndProbs(state, action))

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None
        return max(actions, key=lambda x: self.computeQValueFromValues(state, x))

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
        length = len(self.mdp.getStates())
        for i in range(self.iterations):
            currentState = self.mdp.getStates()[i%length]
            actions = self.mdp.getPossibleActions(currentState)
            if self.mdp.isTerminal(currentState) or not actions:
                self.values[currentState] = 0
            else:
                self.values[currentState] = max(self.computeQValueFromValues(currentState,action) for action in actions)


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
        predecessors = util.Counter()
        queue = util.PriorityQueue()
        for state in self.mdp.getStates():
            predecessors[state] = set()
            queue.update(state,self.diff(state))


        for state in self.mdp.getStates():
            predecessor = state
            actions = self.mdp.getPossibleActions(predecessor)
            for action in actions:
                for successor, prob in self.mdp.getTransitionStatesAndProbs(predecessor,action):
                    if prob > 0:
                        predecessors[successor].add(predecessor)


        for i in range(self.iterations):
            if queue.isEmpty():
                break
            currentState = queue.pop()
            if self.mdp.isTerminal(currentState):
              continue
            self.values[currentState] = self.updateValue(currentState)
            for predecessor in predecessors[currentState]:
                if abs(self.diff(predecessor)) > self.theta:
                    queue.update(predecessor,self.diff(predecessor))


    def diff(self, state):
        return -abs(self.values[state]-self.updateValue(state))

    def updateValue(self,state):
        if self.mdp.isTerminal(state):
            return 0
        return max(self.computeQValueFromValues(state,action) for action in self.mdp.getPossibleActions(state))