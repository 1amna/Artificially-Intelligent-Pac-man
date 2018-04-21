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

INF = 999999


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

		if successorGameState.isWin():
			return float('inf')

		currentScore = currentGameState.getScore()
		nextScore = successorGameState.getScore()
		score = 0

		if not (nextScore < currentScore or nextScore == currentScore):
			score = score + 5000

		nearest = 555555
		index = 0
		foodList = newFood.asList()
		doubleLength = 2 * len(foodList)

		while index < doubleLength:
			foodItem = foodList[index/2]
			gapLength = (manhattanDistance(foodItem, newPos))
			if not (gapLength >= nearest):
				nearest = gapLength
			index = index + 2

		score = score - abs(nearest)
		ghPos = successorGameState.getGhostPosition(1)
		gNumScared = newScaredTimes[0]

		if not ((newPos < ghPos or ghPos < newPos) or
				(gNumScared < 0 or gNumScared > 0)):
			score = - float('inf')

		stopFlag = action.find('Stop')

		if stopFlag == 0:
			score = -5555555

		return score


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
		"""

		minimaxAction = []
		actionList = []
		x = 0
		y = 1
		pac = 0
		i = 0

		while i < len(gameState.getLegalActions(pac)):
			gState = gameState.getLegalActions(pac)[i]
	 		stSuccessor = gameState.generateSuccessor(pac, gState)
			stSucMinMax = self.miniMax(stSuccessor, y, x)
			minimaxAction.append(stSucMinMax)
			actionList.append(gState)
			i = i + 1

		maxAction = max(minimaxAction)
		mActionI = minimaxAction.index(maxAction)
		result = actionList[mActionI]

		return result

	def miniMax(self, gState, agentIndex, depth):
		"""
		  Return minimax value moves for ghosts and pacman involved in
		  gState.
		"""

		agentsCount = gState.getNumAgents()
		allVals = []
		index = 0

		if not (agentsCount < agentIndex or agentsCount > agentIndex):
			depChange = depth + 1
			setAgentIndex = 0
			mMResult = self.miniMax(gState, setAgentIndex, depChange)
			return mMResult

		legalActions = gState.getLegalActions(agentIndex)
		stateLoss = gState.isLose()
		stateWin = gState.isWin()

		if stateWin:
			winEvaluate = self.evaluationFunction(gState)
			return winEvaluate

		elif len(legalActions) < 1:
			emptyEvaluate = self.evaluationFunction(gState)
			return emptyEvaluate

		elif not (self.depth < depth or self.depth > depth):
			sameDEvaluate = self.evaluationFunction(gState)
			return sameDEvaluate

		elif stateLoss:
			lossEvaluate = self.evaluationFunction(gState)
			return lossEvaluate

		while index < len(legalActions):
			lAction = legalActions[index]
			stSuc = gState.generateSuccessor(agentIndex, lAction)
			newIndex = agentIndex + 1
			mM = self.miniMax(stSuc, newIndex, depth)
			allVals.append(mM)
			index = index + 1

		if not (agentIndex == 0):
			minVal = min(allVals)
			return minVal

		elif agentIndex == 0:
			maxVal = max(allVals)
			return maxVal


class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	  Your minimax agent with alpha-beta pruning (question 3)
	"""

	def getAction(self, gameState):
		"""
		  Returns the minimax action using self.depth and self.evaluationFunction
		"""


		initBetaVal = float ('inf')
		initAlphaVal = - float('inf')
		setDepth = 0
		setAgentIndex = 0
		mMActionList = self.maxPruning(gameState, setAgentIndex,  initBetaVal,
				initAlphaVal, setDepth)
		minimaxAction = mMActionList[1]

		return minimaxAction

	def noAction(self, gState):
		"""
		  Return a list of two items including the non-action taken at the
		current	gState.
		"""

		resultList = []
		stateVal = self.evaluationFunction(gState)
		actionTaken = None
		resultList.append(stateVal)
		resultList.append(actionTaken)

		return resultList

	def minPruning(self, gState, agentIndex, betaVal, alphaVal, depth):
		"""
		  Return the result of min value pruning as a list of two items.
		"""

		stateLoss = gState.isLose()
		stateWin = gState.isWin()
		agentsCount = gState.getNumAgents()
		actionTaken = None
		currentVal = float('inf')
		legalActions = gState.getLegalActions(agentIndex)

		if stateWin:
			result = self.noAction(gState)
			return result

		elif len(legalActions) < 1:
			result = self.noAction(gState)
			return result

		elif stateLoss:
			result = self.noAction(gState)
			return result

		elif not (self.depth < depth or self.depth > depth):
			result = self.noAction(gState)
			return result

		index = 0
		while index < len(legalActions):
			legalAction = legalActions[index]
			lessA = agentsCount - 1

			if lessA > agentIndex or agentIndex > lessA:
				newAgentIndex = agentIndex + 1
				nextSucc = gState.generateSuccessor(agentIndex,legalAction)
				val = self.minPruning(nextSucc, newAgentIndex, betaVal,
								alphaVal,  depth)

			elif not (lessA > agentIndex or agentIndex > lessA):
				newDepth = depth + 1
				newAgentIndex = 0
				nextSucc = gState.generateSuccessor(agentIndex,legalAction)
				val = self.maxPruning(nextSucc, newAgentIndex, betaVal,
						alphaVal, newDepth)

			curMVal = val[0]

			if not (currentVal < curMVal or currentVal == curMVal):
				currentVal = curMVal
				actionTaken = legalAction

			if not (currentVal > alphaVal):
				resultList = []
				resultList.append(currentVal)
				resultList.append(actionTaken)
				return resultList

			betaVal = min(betaVal, currentVal)
			index = index + 1

		rList = []
		rList.append(currentVal)
		rList.append(actionTaken)

		return rList

	def maxPruning(self, gState, agentIndex, betaVal, alphaVal, depth):
		"""
		  Return the result of max value pruning as a list of two items.
		"""

		stateLoss = gState.isLose()
		stateWin = gState.isWin()
		actionTaken = None
		currentVal = - float('inf')
		legalActions = gState.getLegalActions(agentIndex)

		if stateWin:
			result = self.noAction(gState)
			return result

		elif len(legalActions) < 1:
			result = self.noAction(gState)
			return result

		elif stateLoss:
			result = self.noAction(gState)
			return result

		elif not (self.depth < depth or self.depth > depth):
			result = self.noAction(gState)
			return result

		index = 0
		while index < len(legalActions):
			legalAction = legalActions[index]
			newAgent = agentIndex + 1
			nextSucc = gState.generateSuccessor(agentIndex, legalAction)
			val = self.minPruning(nextSucc, newAgent, betaVal,  alphaVal,
					depth)
			minPrCurVal = val[0]

			if not(currentVal > minPrCurVal or minPrCurVal == currentVal):
				currentVal = minPrCurVal
				actionTaken = legalAction

			if not(currentVal < betaVal):
				rlist = []
				rlist.append(currentVal)
				rlist.append(actionTaken)
				return rlist

		 	alphaVal = max(alphaVal, currentVal)
		 	index = index + 1

		rList = []
		rList.append(currentVal)
		rList.append(actionTaken)

		return rList


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

		compiledActions = self.compileActions(gameState)
		legalList = compiledActions[1]
		scorelist = compiledActions[0]
		maximum = max(scorelist)
		maxIndex = scorelist.index(maximum)
		actionResult =  legalList[maxIndex]

		return actionResult

	def compileActions(self, gameState):
		"""
		  Return a list containg the lists of scores and legal actions for the
		  give gameState
		"""
		legalActions = []
		score = []
		index = 0
		cur_player = 0
		gameActions = gameState.getLegalActions(cur_player)

		while index < len(gameActions):
			legalAction = gameActions[index]
			nextSucc = gameState.generateSuccessor(cur_player, legalAction)
			nextAgentIndex = cur_player + 1
			nextDepth = 0
			nextScore = self.Expectimax(nextSucc, nextAgentIndex, nextDepth)
			score.append(nextScore)
			legalActions.append(legalAction)
			index = index + 1

		finalList = []
		finalList.append(score)
		finalList.append(legalActions)

		return finalList

	def Expectimax(self, gState, agentIndex, depth):
		"""
		  Return the expectimax values for the game.
		"""
		stateLoss = gState.isLose()
		stateWin = gState.isWin()
		agentsCount = gState.getNumAgents()

		if not (agentsCount> agentIndex or agentsCount < agentIndex):
			newDepth = depth + 1
			curAgentIndex = 0

			return self.Expectimax(gState, curAgentIndex, newDepth)

		legalActions = gState.getLegalActions(agentIndex)

		if stateWin:
			winEvaluate = self.evaluationFunction(gState)
			return winEvaluate

		elif len(legalActions) < 1:
			emptyEvaluate = self.evaluationFunction(gState)
			return emptyEvaluate

		elif not (self.depth < depth or self.depth > depth):
			sameDEvaluate = self.evaluationFunction(gState)
			return sameDEvaluate

		elif stateLoss:
			lossEvaluate = self.evaluationFunction(gState)
			return lossEvaluate

		vals = self.expectimaxValues(gState, agentIndex, depth, legalActions)

		return vals

	def expectimaxValues(self, gState, agentIndex, depth, legalActions):
		"""
		  Calculate the expectimax values given the agentIndex, depth, and a
		  list of legal actions for game state.
		"""
		vals = []
		index = 0

		while index < len(legalActions):
			legalAction = legalActions[index]
			nextSucc = gState.generateSuccessor(agentIndex, legalAction)
			newAgentIndex = agentIndex + 1
			nextVal = self.Expectimax(nextSucc, newAgentIndex, depth)
			vals.append(nextVal)
			index = index + 1

		if agentIndex < 0 or agentIndex > 0:
			total = 0
			valCount = 0

			for val in vals:
				total = total + val
				valCount = valCount + 1

			ExpVal = float(total) / (valCount)

		elif not (agentIndex < 0 or agentIndex > 0):
			ExpVal = max(vals)

		return ExpVal


def betterEvaluationFunction(currentGameState):
	"""
	  Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
	  evaluation function (question 5).

	  DESCRIPTION: Return the optimized score which takes into account the
	  manhattan distance between the ghosts and Pacman, as well as the energy
	  pellets available and Pacman's position. The score gets better when the
	  following cases are satisfied:

	  The pellets are close in promiximity to Pacman.  The score also improves
	  when there are fewer pellets remaining.

	  As for ghosts, the Pacman has to avoid the regular ones, so the longer the
	  manhattan distance between Pacman and each of the ghosts, the better the
	  score becomes. In contrast, the shorter the distance between scared
	  ghosts and Pacman, the better is the score.

	  The chosen values below serve the purpose of drastically representing the
	  conditions aforementioned.

	"""

	pacmanPos = currentGameState.getPacmanPosition()

	stateLoss = currentGameState.isLose()
	stateWin = currentGameState.isWin()

	if stateLoss:
		return -float(INF)

	if stateWin:
		return float(INF)


	initialScore = currentGameState.getScore()

	curScore = 350 + initialScore

	newGhosts = currentGameState.getGhostStates()
	GhostDistanceScore = getGhostDistanceScore(currentGameState, curScore,
			newGhosts, pacmanPos)

	pacmanPellets = currentGameState.getFood()
	latestScore = getPelletsDistanceScore(currentGameState,
	 		GhostDistanceScore, pacmanPellets, pacmanPos)

	return latestScore


def getGhostDistanceScore(gState, curScore, newGhosts, pacmanPos):
	"""
	  Return an updated curScore which takes into account the manhattanDistance
	  position of regular ghosts and scared ghosts compared to Pacman's.
	"""
	index = 0
	while index < len(newGhosts):
		newGhost = newGhosts[index]
		newGhostPos = newGhost.getPosition()
		ManDis = manhattanDistance(pacmanPos, newGhostPos)
		ghostScaredT = newGhost.scaredTimer

		if ghostScaredT >= 1:
			newVal = 90/ManDis
			curScore = curScore + newVal

		if not(ManDis < 1):
			newVal = 2/ManDis
			curScore = curScore - newVal

		index = index + 1

	return curScore


def getPelletsDistanceScore(gState, curScore, pacmanPellets, pacmanPos):
	"""
	Return an updated curScore which takes into account the number of pellets
	remainig as well as the manhattanDistanceposition of the energy pellets
	compared to Pacman's position.
	"""
	pelletDist = 5555
	pelletsList = pacmanPellets.asList()

	index = 0

	while index < len(pelletsList):
		pellet = pelletsList[index]
		ManDis = manhattanDistance(pellet, pacmanPos)

		if not (ManDis >= pelletDist):
			oldPelletDis = pelletDist
			pelletDist = ManDis

		index = index + 1

	curScore = curScore - pelletDist
	manyPellets = len(pelletsList)
	threeMany = 3 * manyPellets
	curScore = curScore - threeMany

	return curScore


# Abbreviation
better = betterEvaluationFunction
