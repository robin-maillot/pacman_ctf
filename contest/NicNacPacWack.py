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
from copy import copy, deepcopy
from capture import SONAR_NOISE_RANGE, SONAR_NOISE_VALUES, SIGHT_RANGE, COLLISION_TOLERANCE


#################
# Team creation #
#################

#def setGlobalVariables(agentIndex)
#    agentValue[agentIndex] = agentIndex
SONAR_MAX = (SONAR_NOISE_RANGE - 1)/2

class enemyAgent(object):
    def __init__(self):
        self.id = 0
        self.startPos = []
        self.grid = []
        self.touchingAgent = [False, False]

    def setEnemyId(self, val):
        self.id = val

    def getEnemyId(self):
        return self.id

    def setGridSize(self, val):
        self.grid = [[0 for i in range(val[1])] for j in range(val[0])]
        #print "val: {}".format(val)

    def setGhostStart(self):
        self.grid = [[0 for i in range(len(self.grid[0]))] for j in range(len(self.grid))]
        self.grid[self.startPos[0]][self.startPos[1]] = 1
        self.touchingAgent = [False, False]

    def setTouchingAgent(self, agent):
        self.touchingAgent[agent] = True

    def notTouchingAgent(self, agent):
        self.touchingAgent[agent] = False


    def updateGridMeasurment(self, selfcopy, gameState, measurement):

        Pos = gameState.getAgentState(selfcopy.index).getPosition()
        #print "gridmeasx: {}".format(len(self.grid[0]))
        #print "gridmeasy: {}".format(len(self.grid))
        #print "Pos: {}".format(Pos)
        #print "measurement: {}".format(measurement)       

        for x in range(0,len(self.grid)):
            #util.pause()
            for y in range(0,len(self.grid[0])):
                #print "x: {}, y: {}".format(x, y)
                #print "ghost: {}, location: [{},{}], value: {}".format(self.id, x, y, self.grid[y][x])
                if self.grid[x][y] > 0:
                    #selfcopy.debugDraw((x, y), [self.grid[x][y],0,self.grid[x][y]],False)
                    #print "distance: {}".format(util.manhattanDistance(Pos, (x,y)))
                    dist = util.manhattanDistance(Pos, (x,y))
                    if abs(dist - measurement) > SONAR_MAX+1: #+1 is magic
                        self.grid[x][y] = 0
                        selfcopy.debugDraw((x, y), [0,0,0],False)
                        #if selfcopy.playerId == 0:
                            #print "enemy: {}, distance: {}, measurement: {}".format(self.id, dist, measurement)
                    #print "distance: {}".format(selfcopy.distancer.getDistance(Pos, (x,y)))
                    #print "distance: {}".format(selfcopy.distancer.getDistanceOnGrid(Pos, (x,y)))
                    #print "distance: {}".format(selfcopy.getMazeDistance(Pos, (x,y)))
        #util.pause()



    def updateGridMotion(self, selfcopy, gameState):
        #print "x: {}".format(len(self.grid))
        #print "y: {}".format(len(self.grid[0]))
        prevgrid = deepcopy(self.grid)

        dx = [1, 0, -1, 0]
        dy = [0, 1, 0, -1]

        walls = gameState.getWalls()
        #print walls
        #print "type: {}".format(getattr(walls))
        
        """for attr_name in dir(walls):
            attr_value = getattr(walls, attr_name)
            print(attr_name, attr_value, callable(attr_value))"""


        """print "wallx : {}".format(len(walls.data))
        print "wally : {}".format(len(walls.data[0]))

        print "akkkkkkk"
        for z in range(0,len(self.grid)):
            print self.grid[z]
        #util.pause()

        print "ahhhhhh" """
        minimum = 9001
        for row in self.grid:
            for i in range(0,len(self.grid[0])):
                if row[i] > 0 and row[i] < minimum:
                    minimum = row[i]
        #print minimum

        for x in range(0,len(self.grid)):
            #util.pause()
            for y in range(0,len(self.grid[0])):
                #print "x: {}, y: {}".format(x, y)
                #print "ghost: {}, location: [{},{}], value: {}".format(self.id, x, y, self.grid[x][y])
                if prevgrid[x][y] > 0:
                    for i in range(0,4):
                        #if x+dx[i] >= len(self.grid) or x+dx[i] < 0 or y+dy[i] >= len(self.grid[0]) or y+dy[i] < 0:
                        if not walls.data[x+dx[i]][y+dy[i]]:
                            try:
                                #if not self.grid[x+dx[i]][y+dy[i]] > 0.75:
                                self.grid[x+dx[i]][y+dy[i]] = minimum
                            except Exception:
                                print "terrible, terrible problem"
                                print "Agent: {}, location: [{},{}], value: {}".format(self.id, x, y, self.grid[x][y])
                                print "[x+dx,y+dy]: [{},{}], isWall: {}".format(x+dx[i], y+dy[i], walls.data[x+dx[i]][y+dy[i]])
                                raise
                                util.pause()
                        else:
                            #print "wall: {}, location: [{},{}], value: {}".format(self.id, x, y, self.grid[x][y])
                            #selfcopy.debugDraw((x+dx[i], y+dy[i]), [1,0,0],False)   
                            continue
        totalsum = 0
        for row in self.grid:
            totalsum += sum(row)
        #print "chugalug"
        #print totalsum
        for row in self.grid:
            if sum(row) > 0:
                row = [float(i)/totalsum for i in row]

    def exactPosition(self, measurement):
        self.grid = [[0 for i in range(len(self.grid[0]))] for j in range(len(self.grid))]
        self.grid[int(measurement[0])][int(measurement[1])] = 1

    def notInSight(self, posi):
        #print "Agent: {}, pos {}".format(selfcopy.index, pos)
        pos = (posi[1],posi[0])
        for i in range(0, SONAR_MAX):
            for j in range(0, SONAR_MAX):
                if i+j < SONAR_MAX:
                    x = int(i)
                    y = int(j)
                    #print "Agent: {}, pos {}, i: {}, j: {}".format(selfcopy.index, pos, i ,j)
                    if i+pos[0] < len(self.grid) and pos[1]+j < len(self.grid[0]):
                        self.grid[int(pos[0]+i)][int(pos[1]+j)] = 0
                    if pos[0]-i <= 0 and pos[1]+j < len(self.grid[0]):
                        self.grid[int(pos[0]-i)][int(pos[1]+j)] = 0
                    if pos[0]-i >= 0 and j-pos[1] >= 0:
                        self.grid[int(pos[0]-i)][int(pos[1]-j)] = 0
                    if pos[0]+i < len(self.grid) and j-pos[1] >= 0:                        
                        self.grid[int(pos[0]+i)][int(pos[1]-j)] = 0


    def drawGrid(self, selfcopy):
        for x in range(0,len(self.grid)):
            #util.pause()
            for y in range(0,len(self.grid[0])):
                #print "x: {}, y: {}".format(x, y)
                #print "ghost: {}, location: [{},{}], value: {}".format(self.id, x, y, self.grid[y][x])
                if self.grid[x][y] > 0:
                    selfcopy.debugDraw((x, y), [1,0,1],False)
        #print self.grid


# A shared memory class, containing a counter and a increment function. 
# This might get weird if you play the same team vs itself. If you want to do that just copy this file and play myteam vs myteamcopy.
class SharedMemory(CaptureAgent):
    # this is the constructor for the class. It gets called wehn you create an instance of the class. Inits counter to 0.
    def __init__(self):
        self.treeAction = [0, 0];
        
        self.enemy = []

        self.enemy.append(enemyAgent())
        self.enemy.append(enemyAgent())

        #print layout.getLayout( options.layout )
        #print self.enemy
        #util.pause()
        
    # returns the state of each pacman
    def setTreeAction(self, agent, act):
        self.treeAction[agent] = act

    def getTreeAction(self, agent):
        return self.treeAction[agent]

    def getEnemy(self, val):
        return enemy[val]


        
# create instance of the class. The "whatever" variable is in the global scope, so it can be accessed from your agents chooseAction function.
sharemem = SharedMemory();



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
'''
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
'''
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
        #default code
        CaptureAgent.registerInitialState(self, gameState)
        self.depth = 2
        self.safe = self.safetyPlaces(gameState)

        for agent in self.getTeam(gameState):
            # Add opponents to list of enemies
            if not agent == self.index:
                agentId = agent

        #I don't know why I have this, but it's a thing. We have ally1 and ally2
        if self.index < agentId:
            self.playerId = 0
        else:
            self.playerId = 1

        #sets enemyID in sharemem
        i = 0
        for agent in self.getOpponents(gameState):
            sharemem.enemy[i].setEnemyId(agent)
            i += 1
        #print "sharemem = {}" .format(sharemem.enemy[0].id)
        #print "sharemem = {}" .format(sharemem.enemy[1].id)

        #finds Start Position in sharemem
        temp = []
        for emory in sharemem.enemy:
            #emory.startPos = gameState.getAgentState(emory.id).getPosition()
            emory.startPos = gameState.getInitialAgentPosition(emory.id)
            #print "startpos for agent {} = {}" .format(emory.id,emory.startPos)
            self.debugDraw([emory.startPos], [0,1,0],False)
        
        #finds the Size of the group, depending if we're red or not (probably a terrible method, but it works)
        gridSize = []

        """if gameState.isRed(gameState.getAgentState(sharemem.enemy[0].id).getPosition()): #if enemy is red, and find top-right ally
            teamId = self.getTeam(gameState)
            print "teamID = {}" .format(teamId)
            if gameState.getAgentState(teamId[0]).getPosition()[1] > gameState.getAgentState(teamId[1]).getPosition()[1]:
                gridSize = gameState.getAgentState(teamId[0]).getPosition()
            else:
                gridSize = gameState.getAgentState(teamId[1]).getPosition()
        else:   #else enemy is blue and find top-right agent
            if gameState.getAgentState(sharemem.enemy[0].id).getPosition()[1] > gameState.getAgentState(sharemem.enemy[1].id).getPosition()[1]:
                gridSize = gameState.getAgentState(sharemem.enemy[0].id).getPosition()
            else:
                gridSize = gameState.getAgentState(sharemem.enemy[1].id).getPosition()"""
        #gameState.getAgentState(emory.id).getPosition()
        walls = gameState.getWalls()
        gridSize = [len(walls.data), len(walls.data[0])]
        #print gridSize       
        sharemem.enemy[0].setGridSize(gridSize)
        sharemem.enemy[1].setGridSize(gridSize)
        sharemem.enemy[0].setGhostStart()
        sharemem.enemy[1].setGhostStart()
        sharemem.enemy[0].updateGridMotion(self, gameState) 
        sharemem.enemy[1].updateGridMotion(self, gameState) 
        #util.pause()





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
        self.debugDraw([goto], [0,1,0],True)
        return dmin
    
    def appxEnemyPos(self, gameState, a):
        if(a==None):
            ghostPositions = map(lambda g: g.getPosition(), ghostStates)
        return 0;


    # Main function
    # Used to calculate all the resulting features from an action.
    # So far takes into account: distance to the border, distance to closest ghost, distance to closest food and distance to closest pacman
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
        friendState = gameState.getAgentState((self.index+2)%4)

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
        #print("ghostpositions: {}".format(ghostPositions))
    #    computeMazeDistances(walls)
    
        # getting closer to food is good
        # getting closer to ghosts is bad
    

        #print("Agent = {} treeAction = {}".format(self.index, sharemem.treeAction[self.playerId]))

        for agent in self.getTeam(newGameState):
            # Add opponents to list of enemies
            if not agent == self.index:
                ally = newGameState.getAgentState(agent)
                agentId = agent


        sharemem.setTreeAction(self.playerId, self.index)
        #print("Agent = {} treeAction = {}".format(self.index, sharemem.treeAction[self.playerId]))



        foodScore = 0
        Pos = myNewState.getPosition()

        if food.asList():
            distanceToClosestFood = min(map(lambda x: self.getMazeDistance(Pos, x), food.asList()))
    
        if(len(ghostPositions)>0):
            distanceToClosestGhost = min(map(lambda x: self.getMazeDistance(Pos, x), 
                                         ghostPositions))
        else:
            distanceToClosestGhost=100
        
        enemyPacmanPossiblePositions = {}
        #Find closest enemy and best position to intercept him 
        for agent in self.getOpponents(newGameState):
            # Add opponents to list of enemies
            enemy = newGameState.getAgentState(agent)
            if(enemy.isPacman and enemy.getPosition() != None):
                enemyPacmanPossiblePositions[agent] = map(lambda a: gameState.generateSuccessor(agent, a),gameState.getLegalActions(agent))
        PacmanFollowing = -1;
        distanceToEnemyPacman = 999
        goTo = None
        for id in enemyPacmanPossiblePositions:
            #print id
            for enemyP in enemyPacmanPossiblePositions[id]:
                if self.getMazeDistance(Pos, enemyP.getAgentPosition(id))<distanceToEnemyPacman:
                    pacmanFollowing = id
                    distanceToEnemyPacman = self.getMazeDistance(Pos, enemyP.getAgentPosition(id))
                    goTo = enemyP.getAgentPosition(id)
        #if(goTo!=None):
        #    self.debugDraw([goTo], [1,0,0],True)
            
        pacmanScore = 0
        ghostScore = 0
        foodScore = 0
        captureScore = 0
        friendScore = 0       
        
        if distanceToEnemyPacman == 0:
            pacmanScore = 2
        elif distanceToEnemyPacman < 999:
            pacmanScore = 1/distanceToEnemyPacman
                
            
        if distanceToClosestGhost == 0:
           ghostScore = -999
        elif distanceToClosestGhost < 6:
          ghostScore = (1./distanceToClosestGhost)
        
        if food.asList():
            if(len(food.asList())==len(oldfood.asList())-1):
                foodScore = 2
            elif distanceToClosestFood == 0:
                foodScore = 0
                ghostScore += 2
            else:
                foodScore = 1./distanceToClosestFood
        if (myOldState.isPacman and myOldState.numCarrying>0):
            d = self.distanceToCamp(newGameState)
            #print(str(d))
            if d==0:
                captureScore = 999
            else:
                captureScore = math.sqrt(myNewState.numCarrying) *1./self.distanceToCamp(newGameState)
                
        if friendState.getPosition()!=None:
            if self.getMazeDistance(Pos, friendState.getPosition())>0:
                friendScore = 1/self.getMazeDistance(Pos, friendState.getPosition())
            else:
                friendScore = 1
        #print(str(a)+":"+str(foodScore)+","+str(ghostScore)+","+str(captureScore)+","+str(myNewState))
        features['foodScore'] = foodScore
        features['ghostScore'] = ghostScore
        features['captureScore'] = captureScore
        features['pacmanScore'] = pacmanScore
        features['friendScore'] = friendScore


        return features
    
    # Define weights for each of the features.
    def getWeights(self, gameState, a):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'foodScore': 1.0,'ghostScore': -2.0,'captureScore': 1.0,'pacmanScore':0.0,'friendScore':-0.0}
    
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

        return {'foodScore': 1.0,'ghostScore': -2.0,'captureScore': 2.0,'pacmanScore':0.0,'friendScore':-1.0}
    
    def updateGridMeasurment(self, gameState):
        #print "self: {}, self+1%4: {}".format(self.index,(self.index+1)%4)
        enemyAfter = 0
        if not sharemem.enemy[enemyAfter].id == (self.index+1)%4:
            enemyAfter =1
        sharemem.enemy[enemyAfter].updateGridMotion(self, gameState)      #first updates motion model

        #sharemem.enemy[(self.index+1)%4].updateGridMotion(self, gameState)      #first updates motion model
        for emy in sharemem.enemy:

            """print "self"
            for attr_name in dir(self):
                attr_value = getattr(self, attr_name)
                print(attr_name, callable(attr_value))
            util.pause()

            print "ded"
            print self.evaluateState()
            for attr_name in dir(self.getPreviousObservation()):
                attr_value = getattr(self.getPreviousObservation(), attr_name)
                print(attr_name, callable(attr_value))
            util.pause()

            
            print "gameState"
            for attr_name in dir(gameState.getAgentState(self.index)):
                attr_value = getattr(gameState.getAgentState(self.index), attr_name)
                print(attr_name, callable(attr_value))
            util.pause()
            


            if gameState.GhostRules().checkDeath(emy.id):
                print"ded dead"
                util.pause()"""

            emy.updateGridMeasurment(self, gameState, gameState.getAgentDistances()[emy.id])   #incorperates noisy sonar measurement
            #print gameState.isScared(self.index)




            if gameState.getAgentState(emy.id).getPosition() != None:       #if we see the enemy agent, update their position
                emy.exactPosition(gameState.getAgentState(emy.id).getPosition())

                if self.getMazeDistance(gameState.getAgentState(emy.id).getPosition(), gameState.getAgentState(self.index).getPosition()) < 1.5:
                    emy.setTouchingAgent(self.playerId)  #first check if agent is touching
                    print "our agent {} is touching {}".format(self.index, emy.id)

            else:
                #print "team: {}, agentnum: {}, index: {}".format(self.getTeam(gameState),agentNum, self.index)
                if emy.touchingAgent[self.playerId]:   #the agent is out of range and has thus jumped 6 sonar levels
                        sizeofmove = util.manhattanDistance(gameState.getAgentState(self.index).getPosition(), self.previousLocation)
                        if not gameState.getAgentState(self.index).isPacman or gameState.getAgentState(emy.id).scaredTimer > 0:
                            if not abs(sizeofmove) >1:
                                print "GET MUNCHED"
                                emy.setGhostStart()
                                emy.updateGridMeasurment(self, gameState, gameState.getAgentDistances()[emy.id]) 
                            else:
                                print "rip"
                                emy.notTouchingAgent(self.playerId)
                else:
                    emy.notTouchingAgent(self.playerId)  #otherwise everything is fine, carry on.
                    emy.notInSight(gameState.getAgentState(self.index).getPosition())

        self.getCapsules(gameState)





        if self.playerId == 0:
            sharemem.enemy[0].drawGrid(self)
            """
            #print "entering gridmeasure"
            #print sharemem.getTreeAction(self.playerId)
            #print gameState.getAgentDistances()     #Noisy Data!!!
            ##print self.getCurrentObservation()
            #print "distancer: ".format(self.distancer.getMazeDistances())#(self.index, sharemem.enemy[0].id))"""
            





    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluateState(gameState,a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        #values = [self.minmax(gameState.generateSuccessor(self.index, a),self.index,0) for a in actions]
        
        self.updateGridMeasurment(gameState)
        self.previousLocation = gameState.getAgentState(self.index).getPosition()
        #print(bestActions)
        
        #print "MakeObservsation1: {}".format(self.getMazeDistance(self.getPosition, enemyP.getAgentPosition(id)))

        #print "MakeObservsation1: {}".format(gameState.makeObservation(sharemem.enemy[1].id))



        #print(shareMemory.getEnemy(0).id)
                
        #minimax(self, gameState, agentIndex, depth)
        #print "waka"

        return random.choice(bestActions)