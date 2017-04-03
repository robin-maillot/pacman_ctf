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
from util import manhattanDistance
import math
import random, util
from game import Agent

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
    first = 'FrenchCanadianAgent', second = 'FrenchCanadianAgent'):
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

class ReflexAgent(CaptureAgent):
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
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    
    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    walls = currentGameState.getWalls()
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates() 
    ghostPositions = map(lambda g: g.getPosition(), newGhostStates)
#    computeMazeDistances(walls)

    # getting closer to food is good
    # getting closer to ghosts is bad

    foodScore = 0
#    distanceToClosestFood = min(map(lambda x: getDistanceInMaze(newPos, x), oldFood.asList()))
    distanceToClosestFood = min(map(lambda x: util.manhattanDistance(newPos, x), oldFood.asList()))

    distanceToClosestGhost = min(map(lambda x: util.manhattanDistance(newPos, x), 
                                     ghostPositions))

    ghostScore = 0
    foodScore = 0
    if distanceToClosestGhost == 0:
      return -99
    elif distanceToClosestGhost < 6:
      ghostScore = (1./distanceToClosestGhost) * -2
    
    if distanceToClosestFood == 0:
      foodScore = 0
      ghostScore += 2
    else:
      foodScore = 1./distanceToClosestFood

    return foodScore + ghostScore

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
    
    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()


class MultiAgentSearchAgent(CaptureAgent):
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
        CaptureAgent.registerInitialState(self, gameState)
        self.depth = 2
        self.safe = self.safetyPlaces(gameState)

        
    def safetyPlaces(self,gameState):
        safetyCoordinates = []
        x = gameState.data.layout.width/2
        ymax = gameState.data.layout.height
        if(self.red):
            x-=1
        for y in range(1,ymax-1):
            if(not gameState.hasWall(x,y)):
                safetyCoordinates.append((x,y))
                self.debugDraw([[x,y]], [0,0,1])
        return safetyCoordinates
            
    def distanceToCamp(self,gameState):
        dmin = 999
        Pos = gameState.getAgentState(self.index).getPosition()
        goto = Pos
        for x in self.safe:
            if(self.getMazeDistance(Pos, x)<dmin):
                dmin=self.getMazeDistance(Pos, x)
                goto=x
        self.debugDraw([goto], [0,1,0])
        return dmin
    
    def getFeatures(self, gameState, a):
        """
        Returns a counter of features for the state
        """

        features = util.Counter()
        if(a==None):
            newGameState = gameState
        else:
            newGameState = gameState.generateSuccessor(self.index, a)
        myOldState = gameState.getAgentState(self.index)
        myNewState = newGameState.getAgentState(self.index)

        enemies = []
        for agent in self.getOpponents(newGameState):
            # Add opponents to list of enemies
            enemies.append(newGameState.getAgentState(agent))
        ghostStates = []
        # Check enemies...
        for enemy in enemies:
            # If there is an enemy position that we can see...
            if not enemy.isPacman and enemy.getPosition() != None:
                # Add that enemy to the list of defenders
                ghostStates.append(enemy)
                
        oldfood = self.getFood(gameState)
        food = self.getFood(newGameState)
        #ghostStates = self.getGhostStates(gameState) 
        ghostPositions = map(lambda g: g.getPosition(), ghostStates)
    #    computeMazeDistances(walls)
    
        # getting closer to food is good
        # getting closer to ghosts is bad
    
        foodScore = 0
        Pos = myNewState.getPosition()
        distanceToClosestFood = min(map(lambda x: self.getMazeDistance(Pos, x), food.asList()))
    
        if(len(ghostPositions)>0):
            distanceToClosestGhost = min(map(lambda x: self.getMazeDistance(Pos, x), 
                                         ghostPositions))
        else:
            distanceToClosestGhost=100
    
        ghostScore = 0
        foodScore = 0
        captureScore = 0
        if distanceToClosestGhost == 0:
          return -99
        elif distanceToClosestGhost < 6:
          ghostScore = (1./distanceToClosestGhost) * -2
        
        if(len(food.asList())==len(oldfood.asList())-1):
            foodScore = 2
        elif distanceToClosestFood == 0:
          foodScore = 0
          ghostScore += 2
        else:
          foodScore = 1./distanceToClosestFood
        if (myOldState.isPacman and myOldState.numCarrying>0):
            d = self.distanceToCamp(newGameState)
            print(str(d))
            if d==0:
                captureScore = 999
            else:
                captureScore = math.sqrt(myNewState.numCarrying) *1./self.distanceToCamp(newGameState)
        print(str(a)+":"+str(foodScore)+","+str(ghostScore)+","+str(captureScore)+","+str(myNewState))
        features['foodScore'] = foodScore
        features['ghostScore'] = ghostScore
        features['captureScore'] = captureScore
        return features
    
    def getWeights(self, gameState, a):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """

        return {'foodScore': 1.0,'ghostScore': 1.0,'captureScore': 1.0}
    
    def evaluateState(self,gameState,a):
        features = self.getFeatures(gameState, a)
        weights = self.getWeights(gameState, a)
        return features * weights
    
    def isWon(self,gameState):
        return (self.getFood(gameState)<=2)
    
    def isLost(self,gameState,enemies):
        defenders = []
        distances_to_defenders = []
        current_position = gameState.getAgentState(self.index).getPosition()
        # Check enemies...
        for enemy in enemies:
            # If there is an enemy position that we can see...
            if not enemy.isPacman and enemy.getPosition() != None:
                # Add that enemy to the list of defenders
                defenders.append(enemy)
    
        # If there is a defender...
        if len(defenders) > 0:
            # Check the indices of defenders...
            for d in defenders:
                # Find the shortest distance to the defender from current position and add to list of defender distances
                distances_to_defenders.append(self.getMazeDistance(current_position, d.getPosition()))
            return (self.getFoodYouAreDefending(gameState)<=2 or min(distances_to_defenders)<1)
        else:
            return (self.getFoodYouAreDefending(gameState)<=2)


class FrenchCanadianAgent(MultiAgentSearchAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

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
        MultiAgentSearchAgent.registerInitialState(self, gameState)
        
        '''
        Your initialization code goes here, if you need any.
        '''

    def gameOver(self, gameState, d):
        enemies = []
        # Check the indices of the opponents...
        for agent in self.getOpponents(gameState):
            # Add opponents to list of enemies
            enemies.append(gameState.getAgentState(agent))
        return self.isLost(gameState,enemies) or self.isWon(gameState) or d == 0
    
    
    def minmax(self, gameState, agentIndex, depth):
        "produces the min or max value for some game state and depth; depends on what agent."
        successorStates = map(lambda a: gameState.generateSuccessor(agentIndex, a),gameState.getLegalActions(agentIndex))
        if self.gameOver(gameState, depth): # at an end
            return self.evaluateState(gameState,None)
        else:
            # use modulo so we can wrap around, get vals of leaves
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            vals = map(lambda s: self.minmax(s, nextAgent, depth - 1),successorStates)      
            if nextAgent == 0: # pacman
                return max(vals)
            else:
                return min(vals)
    
    def getWeights(self, gameState, a):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """

        return {'foodScore': 1.0,'ghostScore': 1.0,'captureScore': 1.0}
    
    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluateState(gameState,a) for a in actions]
        #values = [self.minmax(gameState.generateSuccessor(self.index, a),self.index,0) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        print(bestActions)
        #minimax(self, gameState, agentIndex, depth)
        '''
        You should change this in your own agent.
        '''
        #util.pause()
        return random.choice(bestActions)