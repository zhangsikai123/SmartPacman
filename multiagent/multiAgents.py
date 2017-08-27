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
import search
from game import Agent
from searchAgent import PositionSearchProblem

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        dis = 0xfffffff
        for ghost in newGhostStates:
            curdis = mazeDistance(ghost.getPosition(), newPos,currentGameState)
            dis = min(dis, curdis)
            if ghost.scaredTimer>0 and ghost.scaredTimer<40 and curdis<10:
                return -curdis
        "*** YOUR CODE HERE ***"
        foodDis = 0xfffffff
        for food in newFood.asList():
            if newFood[food[0]][food[1]] is True:
                foodDis = min(foodDis, mazeDistance(newPos,food,currentGameState))
        if len(newFood.asList()) < len(currentGameState.getFood().asList()):
            foodDis = 0
        if dis<4:
            return dis
        return float(10)/(float(foodDis)+1)

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    infinity = 0xfffffff

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
        v = -self.infinity
        legalMoves = gameState.getLegalActions(0)
        move = legalMoves[0]
        for action in legalMoves:
            state = gameState.generateSuccessor(0, action)
            thisV = self.value(state,0,1)
            if thisV > v:
                v = thisV
                move = action
        return move

    def value(self,gameState,depth,index):
        if index == gameState.getNumAgents():
            index = 0
            depth +=1
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if index==0:
            v = -self.infinity
            legalMoves = gameState.getLegalActions(index)
            for action in legalMoves:
                state = gameState.generateSuccessor(index,action)
                v = max(v,self.value(state,depth,index+1))
            return v
        else:
            v = self.infinity
            legalMoves = gameState.getLegalActions(index)
            for action in legalMoves:
                state = gameState.generateSuccessor(index, action)
                v = min(v, self.value(state, depth, index + 1))
            return v
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    infinity = 0xfffffff
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        v = -self.infinity
        legalMoves = gameState.getLegalActions(0)
        move = legalMoves[0]
        for action in legalMoves:
            state = gameState.generateSuccessor(0, action)
            thisV = self.value(state, 0, 1,v,self.infinity)
            if thisV > v:
                v = thisV
                move = action
        return move

    def value(self, gameState, depth, index, alpha, beta):
        if index == gameState.getNumAgents():
            index = 0
            depth += 1
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if index == 0:
            v = -self.infinity
            legalMoves = gameState.getLegalActions(index)
            for action in legalMoves:
                state = gameState.generateSuccessor(index, action)
                v = max(v, self.value(state, depth, index + 1,alpha, beta))
                if v>beta:
                    return v
                alpha = max(alpha,v)
            return v
        else:
            v = self.infinity
            legalMoves = gameState.getLegalActions(index)
            for action in legalMoves:
                state = gameState.generateSuccessor(index, action)
                v = min(v, self.value(state, depth, index + 1,alpha,beta))
                if v < alpha:
                    return v
                beta = min(beta,v)
            return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    infinity = 0xfffffff

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
        v = -self.infinity
        legalMoves = gameState.getLegalActions(0)
        move = legalMoves[0]
        for action in legalMoves:
            state = gameState.generateSuccessor(0, action)
            thisV = self.value(state, 0, 1)
            if thisV > v:
                v = thisV
                move = action
        return move

    def value(self, gameState, depth, index):
        if index == gameState.getNumAgents():
            index = 0
            depth += 1
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if index == 0:
            v = -self.infinity
            legalMoves = gameState.getLegalActions(index)
            for action in legalMoves:
                state = gameState.generateSuccessor(index, action)
                v = max(v, self.value(state, depth, index + 1))
            return v
        else:
            v = 0
            legalMoves = gameState.getLegalActions(index)
            for action in legalMoves:
                state = gameState.generateSuccessor(index, action)
                v += self.value(state, depth, index + 1)
            return float(v)/float(len(legalMoves))

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    pos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    capsuleWeight = 1
    ghostAvoidWeight = 0.5
    foodDistanceWeight = 1
    scoreWeight = 1
    eatFoodWeight = 10
    ghosts = [ghost.getPosition() for ghost in currentGameState.getGhostStates()]
    eatFood = len(currentGameState.getFood().asList())
    capsuleDistance = distance(currentGameState.getCapsules(),pos)
    dis = 0xfffffff
    goal = (0,0)
    for food in currentGameState.getFood().asList():
        if manhattanDistance(pos,food) < dis:
            goal = food
    prob = PositionSearchProblem(currentGameState, start=pos, goal = goal, warn=False, visualize=False)
    foodDistance = len(search.bfs(prob))
    ghostDistance = distance(ghosts,pos)
    unStoppable = True

    for ghostState in ghostStates:
        if ghostState.scaredTimer ==0:
            unStoppable = False
            break
    if unStoppable:
        ghostAvoidWeight = -100

    return 25.0/(foodDistance+1) +1000.0/(eatFood+1)+ 20.0/(capsuleDistance+1)+ 2000.0/(len(currentGameState.getCapsules())+1) + ghostAvoidWeight * ghostDistance

# Abbreviation
better = betterEvaluationFunction


def distance(list,pos):
    if len(list) == 0:
        return 0
    newList = [manhattanDistance(pos,e) for e in list]
    return min(newList)
def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[int(x1)][int(y1)], 'point1 is a wall: ' + str(point1)
    assert not walls[int(x2)][int(y2)], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))