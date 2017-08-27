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

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    closed = []
    route = []
    route.append((problem.getStartState(),'Stop',0))
    fringe = util.Queue()
    fringe.push(route)
    while not fringe.isEmpty():
        route = fringe.pop()
        cur = route[len(route)-1] #expand the node at the tail of fringe
        if problem.isGoalState(cur[0]):
            return [x[1] for x in route[1:]]
        if cur[0] not in closed:
            closed.append(cur[0])
            successors = problem.getSuccessors(cur[0])
            for successor in successors:
                newRoute = list(route)
                newRoute.append(successor)
                fringe.push(newRoute)
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0
def aStarSearch(problem, heuristic=nullHeuristic):
    closedset = []
    fringe = util.PriorityQueue()
    start = problem.getStartState()
    fringe.push( (start, []), heuristic(start, problem))

    while not fringe.isEmpty():
        node, actions = fringe.pop()

        if problem.isGoalState(node):
            return actions

        if not node in closedset:
            closedset.append(node)
            for coord, direction, cost in problem.getSuccessors(node):
                new_actions = actions + [direction]
                score = problem.getCostOfActions(new_actions) + heuristic(coord, problem)
                fringe.push( (coord, new_actions), score)

    return []

bfs = breadthFirstSearch
astar = aStarSearch