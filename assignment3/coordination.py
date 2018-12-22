import numpy as np 
import matplotlib.pyplot as plt
import random 
import math

AGENTS_NUMBER = 26
ACTIONS_NUMBER = [2,3] 
ALPHA = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] 

class Agent:
	def __init__(self, agent_ID, indexNbActions):
		self.agent_ID = agent_ID
		self.neighbours = []
		self.action = random.randint(1,ACTIONS_NUMBER[indexNbActions])

	def getNeighbours(self):
		return self.neighbours

	def addNeighbour(self, agent):
		self.neighbours.append(agent)

	def getIndex(self):
		return self.agent_ID

	def getAction(self):
		return self.action 

	def setAction(self, newAction):
		self.action = newAction

	def __repr__(self):
		return str(self.agent_ID)

	def __str__(self):
		return str(self.agent_ID)

#########################################################################################

def initRingTopology(indexNbActions):
	agentsNetwork = []
	for i in range(AGENTS_NUMBER):
		newAgent = Agent(i, indexNbActions)
		agentsNetwork.append(newAgent)

	for j in range(AGENTS_NUMBER): 
		if j == 0:
			agentsNetwork[j].addNeighbour(agentsNetwork[-1])
			agentsNetwork[j].addNeighbour(agentsNetwork[1])
		elif j == AGENTS_NUMBER-1:
			agentsNetwork[j].addNeighbour(agentsNetwork[-2])
			agentsNetwork[j].addNeighbour(agentsNetwork[0])
		else:  
			agentsNetwork[j].addNeighbour(agentsNetwork[j-1])
			agentsNetwork[j].addNeighbour(agentsNetwork[j+1])

	return agentsNetwork

def createPayoffMatrix(indexNbActions):
	payoffMatrix = [[0 for j in range(ACTIONS_NUMBER[indexNbActions])] for i in range(ACTIONS_NUMBER[indexNbActions])]
	
	i = 0 
	for row in payoffMatrix:
		row[i] = 1 
		i += 1 

	return payoffMatrix

def mainLoop(agentsNetwork, payoffMatrix, alphaIndex, Tmax=50000):
	conventionReachedTimes = 0 
	hasConverged = False 
	t = 0 
	while (conventionReachedTimes != 10) and (t <= Tmax):
		# Get random agent and neighbour of the agent
		randomAgent = np.random.choice(agentsNetwork)
		randomNeighbour = np.random.choice(randomAgent.neighbours)

		# Get the actions of each agent 
		oldActionAgent = randomAgent.getAction()
		actionNeighbour = randomNeighbour.getAction()

		# Get the payoffs of the actions 
		payoffAgent = payoffMatrix[oldActionAgent-1][actionNeighbour-1]
		payoffNeighbour = payoffMatrix[actionNeighbour-1][oldActionAgent-1]

		# Update the actions
		actionAgent = updateAction(payoffAgent, oldActionAgent, actionNeighbour, alphaIndex)
		actionNeighbour = updateAction(payoffNeighbour, actionNeighbour, oldActionAgent, alphaIndex)
		randomAgent.setAction(actionAgent)
		randomNeighbour.setAction(actionNeighbour)

		hasConverged = isConventionReached(agentsNetwork)
		if(hasConverged):
			conventionReachedTimes += 1

		t += 1 

	return t 

def updateAction(payoff_i, action_i, action_j, alphaIndex):
	pi_i = max(ALPHA[alphaIndex] - payoff_i, 0)
	if random.random() < pi_i:
		action_i = action_j
	return action_i

def isConventionReached(agentsNetwork):
	actionsList = [agentsNetwork[i].getAction() for i in range(AGENTS_NUMBER)]
	if all(action == actionsList[0] for action in actionsList):
		return True
	else: 
		return False


if __name__ == "__main__":

	convergingTimes = []
	
	for actionIndex in range(len(ACTIONS_NUMBER)):
		convergingMean = [50000]
		for alphaIndex in range(len(ALPHA)): 

			# Tried some executions and for alpha = 0 and 1, it never converges, this condition is just useful to make the run faster 
			# and skipping those cases to avoid to reexcute 200x the loop of 50000
			if ALPHA[alphaIndex] == 0 or ALPHA[alphaIndex] == 1:
				convergingTimes.append(50000)
			else:
				for _ in range(200):
					agentsNetwork = initRingTopology(actionIndex)
					payoffs = createPayoffMatrix(actionIndex)
					timeConverging = mainLoop(agentsNetwork, payoffs, alphaIndex)
					convergingTimes.append(timeConverging)
					print("Converged at time", timeConverging)

				convergingMean.append(np.mean(convergingTimes))
				convergingTimes = []

		print(convergingMean)
		convergingMean.append(50000)
		plt.plot(convergingMean, label="{} actions".format(ACTIONS_NUMBER[actionIndex]))

	#plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [8664.92039800995, 4566.87, 3780.26, 2485.685, 2173.13, 2026.595, 2098.34, 2265.085, 3329.99], label="2 actions")
	#plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [11936.940594059406, 5201.59, 3601.425, 3208.995, 2787.81, 2326.38, 2270.1, 2625.88, 3713.315], label="3 actions")
	plt.xticks(np.arange(11), ALPHA)

	plt.title("Average convergence time for each alpha value")
	plt.ylabel("Average convergence time")
	plt.xlabel("Alpha")

	plt.legend()
	plt.show()


