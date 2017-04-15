# myTeam.py
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
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

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
  def getEnemyPos(self, gameState):
    return [gameState.getAgentPosition(idx) for idx in self.getOpponents(gameState) if gameState.getAgentPosition(idx) != None]

  def connectedCount(self, gameState, obstaclePos, pos):
    def toIntPos(pos):
      return (int(pos[0]), int(pos[1]))
    pos = toIntPos(pos)
    obstaclePos = [toIntPos(p) for p in obstaclePos]
    def dfs(grid, pos):
      def inRange(x, y):
        return 0 <= x and x <= grid.width and 0 <= y and y <= grid.height

      x, y = pos
      vis[x][y] = True
      conn = 1
      for dir in dirs:
        dx, dy = dir
        nx, ny = x + dx, y + dy
        nPos = (nx, ny)
        if inRange(nx, ny) and not vis[nx][ny] and not grid.isWall(nPos) and not nPos in obstaclePos:
          conn += dfs(grid, (nx, ny))
      return conn

    dirs = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    vis = []
    for i in range(0, gameState.data.layout.width):
      vis.append([False] * gameState.data.layout.height)
    print len(vis)
    return dfs(gameState.data.layout, pos)



  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    #
    # print self.connectedCount(gameState, self.getEnemyPos(gameState), gameState.data.layout.getRandomLegalPosition())
    CaptureAgent.registerInitialState(self, gameState)
    layout = gameState.data.layout
    self.cells = []
    self.cnt = 0
    self.belief = [util.Counter() for i in range(4)]

    for i in range(0, layout.width):
      for j in range(0, layout.height):
        if (not layout.isWall((i, j))):
          self.cells += [(i, j)]
    print self.cells

  def getMyPos(self, gameState):
    return gameState.getAgentState(self.index).getPosition()

  def resetPrior(self, opponent):
    for cell in self.cells:
      self.belief[opponent][cell] = 1.0 / len(self.cells)


  def updateNB(self, gameState, agentPos, reading, opponent):
    belief = self.belief[opponent]
    smoothing = 0.000001
    sum = smoothing * len(self.cells)
    for cell in self.cells:
      prior = belief[cell]
      belief[cell] = (gameState.getDistanceProb(util.manhattanDistance(cell, agentPos), reading) + smoothing) * prior
      sum += belief[cell]
    print sum
    print sum
    for cell in self.cells:
      belief[cell] /= sum

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    observableEnemy = [idx for idx in self.getOpponents(gameState) if
                       gameState.getAgentPosition(idx) != None and util.manhattanDistance(gameState.getAgentPosition(self.index), gameState.getAgentPosition(idx)) <= 5]
    players = sorted([self.index] + observableEnemy)
    playerIdx = players.index(self.index)
    bestValue, bestAction = self.alphaBeta(gameState, -float('inf'), float('inf'), playerIdx, players, 6 if len(observableEnemy) > 0 else 4)
    print bestValue, bestAction
    self.cnt = self.cnt + 1

    for opponent in self.getOpponents(gameState)[0:1]:
      self.resetPrior(opponent)

      for observation in self.observationHistory[-4:]:
        self.updateNB(gameState, self.getMyPos(gameState), observation.agentDistances[opponent], opponent)
    # self.getPreviousObservation() and self.updateNB(gameState, self.getMyPos(gameState), self.getPreviousObservation().agentDistances[self.getOpponents(gameState)[0]])
    self.displayDistributionsOverPositions(self.belief)
    return bestAction


  def evaluate(self, gameState):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState)
    weights = self.getWeights(gameState)
    return features * weights

  def closeEnough(self, gameState, opponentIdx):
    return util.manhattanDistance(gameState.getAgentState(self.index).getPosition(), gameState.getAgentPosition(opponentIdx)) <= 5

  def getFeatures(self, gameState):
    """
    Returns a counter of features for the state
    """
    foodList = self.getFood(gameState).asList()
    enemyFoodList = self.getFoodYouAreDefending(gameState).asList()
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]

    myPos = gameState.getAgentState(self.index).getPosition()

    features = util.Counter()
    features['eatenFood'] = -len(foodList)

    features['enemyEatenFood'] = -len(enemyFoodList)

    # get nearest food
    if len(foodList) > 0:
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    # get nearest observable enemy
    oppDist = [self.getMazeDistance(gameState.getAgentPosition(idx), myPos) for idx in self.getOpponents(gameState) if gameState.getAgentPosition(idx) != None]
    if len(oppDist) > 0:
      features['inverseDistanceToOpponent'] = 1.0 / min(oppDist)

    # check if respawned or not
    # todo: make the checking more robust
    isKilled = self.getMazeDistance(myPos, gameState.getInitialAgentPosition(self.index)) < 4
    features['killed'] = 1 if isKilled else 0

    # get nearest friend
    friendDist = [self.getMazeDistance(gameState.getAgentPosition(idx), myPos) for idx in self.getTeam(gameState) if gameState.getAgentPosition(idx) != None and idx != self.index]
    if len(friendDist) > 0:
      features['inverseDistanceToFriend'] = 1.0 / (min(friendDist) + 0.01)

    # get closest distance to border
    borderX = gameState.data.layout.width / 2 + (-1 if gameState.isOnRedTeam(self.index) else 0)
    features['distanceToHome'] = min([self.getMazeDistance(myPos, (borderX, y)) for y in range(0, gameState.data.layout.height) if not(gameState.data.layout.isWall((borderX, y)))])

    # get num of invaders
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    # print len(invaders)

    # get is trapped or not
    # isTrapped = self.connectedCount(gameState, self.getEnemyPos(gameState), myPos)
    # print isTrapped

    # get enmey prob
    enemyProb = 0.5 * sum([self.belief[enemy][myPos] for enemy in self.getOpponents(gameState)])
    # print enemyProb
    return features

  def getWeights(self, gameState):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    def canEatOpp(agentState):
      return not(agentState.isPacman) and agentState.scaredTimer == 0
    myAgentState = gameState.getAgentState(self.index)

    def nearOpp():
        return [idx for idx in self.getOpponents(gameState) if gameState.getAgentPosition(idx) != None and self.getMazeDistance(gameState.getAgentPosition(self.index), gameState.getAgentPosition(idx)) <= 3]
    # print myAgentState.numCarrying
    return {
      'eatenFood': 50.0,# if len(nearOpp()) == 0 or canEatOpp(myAgentState) else 5,
      'enemyEatenFood': -50.0,
      'distanceToFood': -1,
      'inverseDistanceToOpponent': 50 if canEatOpp(myAgentState) else -50,
      'killed': -500,
      'inverseDistanceToFriend': -5,
      'distanceToHome': -0.9 * myAgentState.numCarrying if myAgentState.isPacman else 0,
      'numInvaders': -100,
      'enemyProb': -10
    }

  def alphaBeta(self, gameState, alpha, beta, playerIdx, players, depth):
    # playerIdx is internal idx in players list, not global index
    def nextPlayerIdx():
      return (playerIdx + 1) % len(players)

    if depth == 0:
      return self.evaluate(gameState), None

    player = players[playerIdx]
    if gameState.isOnRedTeam(player) == gameState.isOnRedTeam(self.index):
      # is max player
      v = -float('inf')
      bestAction = None
      for action in gameState.getLegalActions(player):
        actionValue = self.alphaBeta(gameState.generateSuccessor(player, action), alpha, beta, nextPlayerIdx(), players, depth - 1)[0]
        if v < actionValue:
          v = actionValue
          bestAction = action
        alpha = max(alpha, v)
        if beta <= alpha:
          break
      return (v, bestAction)
    else:
      # is min player
      v = float('inf')
      bestAction = None
      for action in gameState.getLegalActions(player):
        actionValue = self.alphaBeta(gameState.generateSuccessor(player, action), alpha, beta, nextPlayerIdx(), players, depth - 1)[0]
        if v > actionValue:
          v = actionValue
          bestAction = action
        beta = min(beta, v)
        if beta <= alpha:
          break
      return (v, bestAction)