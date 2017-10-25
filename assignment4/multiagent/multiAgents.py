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
import time

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

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

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


class StateHelper:
    def utility(self, state):
        """
        simply returns the score of the state
        renamed simply to comply with the textbook algorithm
        """
        return state.getScore()

    def terminal_test(self, state):
        """
        as above, used as syntactic sugar
        used in both minimax and alphabeta to determine a finite state
        """
        return state.isWin() or state.isLose()

    def actions(self, state, agent=0):
        """
        defaults to Pacman as agent,
        as this function is not available to multiagentsearch agents
        (only in the GameState class in pacman.py)
        """
        return state.getLegalActions(agent)

    def max_score(self, action_list, depth):
        """
        takes in a list of (score, action)-tuples
        extracts the highest value (by score)
        returns score if the depth is not 0
        returns the next action otherwise
        """
        best = max(action_list)
        return best[0] if depth else best[1]

    def next_agent(self, state, agent):
        """
        given 2 ghosts and pacman (3 agents), if on agent 2:
        2 == (length=3)-1, which means the next one up will be pacman (0)
        return a tuple of type (int, boolean) <=> (agent, is_pacman)
        """
        return 0 if agent == state.getNumAgents() - 1 else agent+1


class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        s = StateHelper()  # a helper for state-realted computations
        def max_value(state, depth=0):
            if s.terminal_test(state): return s.utility(state)
            action_list = list()
            for a in s.actions(state):
                """
                next state after a move from pacman,
                always the starting point
                """
                result = state.generateSuccessor(0, a)
                """
                check the min value of the next agent (1)
                1 due to it having to be at least 1 ghost for us to calculate
                a min function, otherwise this would be pointless
                """
                min_val_next = min_value(result, depth, 1)
                # append (value,action) to action list
                action_list.append((min_val_next, a))
            # see line 134
            # return if depth > 0 return value, else action
            return s.max_score(action_list, depth)

        def min_value(state, depth, agent):
            if s.terminal_test(state): return s.utility(state)
            agent_next = s.next_agent(state, agent)
            action_list = list()
            for a in s.actions(state, agent):
                # next state after a move from the current agent
                result = state.generateSuccessor(agent, a)
                if agent_next:
                    """
                    this is a ghost (not 0/pacman)
                    we want to find the minimum value
                    i.e. the worst option for pacman
                    """
                    action_list.append(min_value(result, depth, agent_next))
                else:
                    """
                    if: done checking depths? return the score
                    e.g.
                             root           depth = 0
                            /   \
                          1       3         depth = 1, this is the last check!
                         / \     / \
                        1   2   3   4       depth = 2

                    else: not at the nearest depth from pacman,
                    keep getting the highest value move
                    """
                    if depth == self.depth - 1:
                        action_list.append(self.evaluationFunction(result))
                    else:
                        action_list.append(max_value(result, depth + 1))
            return min(action_list)
        return max_value(state=gameState)


class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        s = StateHelper()  # define a statehelper as above
        def max_value(state, depth, alpha, beta):
            if s.terminal_test(state): return s.utility(state)
            v = -float('inf')
            next_action = None
            for a in s.actions(state):
                """
                next state after a move from pacman,
                always the starting point
                """
                result = state.generateSuccessor(0, a)
                """
                check the min value of the next agent (1)
                1 due to it having to be at least 1 ghost for us to calculate
                a min function, otherwise this would be pointless

                instead of using v = max(v, min_result)
                compare them as we need to store the action
                """
                min_result = min_value(result, depth, 1, alpha, beta)
                if min_result > v:
                    next_action = a
                    v = min_result
                if v > beta:
                    break
                alpha = max(alpha, v)
            # see line 134
            # return if depth > 0 return value, else action
            return v if depth else next_action

        def min_value(state, depth, agent, alpha, beta):
            if s.terminal_test(state): return s.utility(state)
            agent_next = s.next_agent(state, agent)
            v = float('inf')
            for a in s.actions(state, agent):
                # next state after a move from the current agent
                result = state.generateSuccessor(agent, a)
                if agent_next:
                    """
                    this is a ghost (not 0/pacman)
                    we want to find the minimum value
                    i.e. the worst option for pacman
                    """
                    tmp = min_value(result, depth, agent_next, alpha, beta)
                else:
                    """
                    if: done checking depths? return the score
                    e.g.
                             root           depth = 0
                            /   \
                          1       3         depth = 1, this is the last check!
                         / \     / \
                        1   2   3   4       depth = 2

                    else: not at the nearest depth from pacman,
                    keep getting the highest value move
                    """
                    if depth == self.depth - 1:
                        tmp = self.evaluationFunction(result)
                    else:
                        tmp = max_value(result, depth + 1, alpha, beta)

                v = min(v, tmp)
                if v < alpha:
                    break
                beta = min(beta, v)
            return v
        inf = float('inf')
        return max_value(state=gameState, depth=0, alpha=-inf, beta=inf)

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
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
