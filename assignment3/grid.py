import numpy as np 
import matplotlib.pyplot as plt
import random 
import math

class Windy_Gridworld:
	"""
	Class doing an instance of Windy Gridworld 
	"""
	def __init__(self):
		self.grid = [[0,0,1,1,1,2,2,1,0,0,0,0],
					[0,0,1,1,1,2,2,1,0,0,0,0],
					[0,0,1,1,1,2,2,1,0,0,0,0],
					[0,0,1,1,1,2,2,1,0,0,0,0],
					[0,0,1,1,1,2,2,1,0,0,0,0],
					[0,0,1,1,1,2,2,1,0,0,0,0],
					[0,0,1,1,1,2,2,1,0,0,0,0]]

		self.currentPos = [3,0]  # [line, col] 3,0 
		self.goal = [3,9]
		self.rowsNb = 7 
		self.colsNb = 12
		self.actionsIndex = {0:"W", 1:"NW", 2:"N", 3:"NE", 4:"E", 5:"SE", 6:"S", 7:"SW"}

	def getCurrentPos(self):
		return self.currentPos

	def setCurrentPos(self,newCurrentPos):
		self.currentPos = newCurrentPos 

	def getGoal(self):
		return self.goal
		
	def setGoal(self,newGoal):
		self.goal = newGoal 

	def getWindValue(self):
		return self.grid[self.currentPos[0]][self.currentPos[1]]

	def windVariation(self, wind):
		return np.random.choice([wind-1, wind, wind+1], p=[1/3, 1/3, 1/3])

	def move(self, direction): 

		# If there's wind in current pos 
		wind = self.getWindValue()
		if wind:
			wind = self.windVariation(wind)
			self.currentPos[0] -= wind 

		# Manage 8 directions 
		if direction == "W": 
			self.currentPos[1] -= 1
		elif direction == "NW":
			self.currentPos[0] -= 1
			self.currentPos[1] -= 1
		elif direction == "N":
			self.currentPos[0] -= 1 
		elif direction == "NE": 
			self.currentPos[0] -= 1
			self.currentPos[1] += 1
		elif direction == "E":
			self.currentPos[1] += 1
		elif direction == "SE":
			self.currentPos[0] += 1
			self.currentPos[1] += 1 
		elif direction == "S":
			self.currentPos[0] += 1 
		elif direction == "SW":
			self.currentPos[0] += 1
			self.currentPos[1] -= 1 

		# Manage if going outside 
		if self.currentPos[0] < 0:  # Outside from the top 
			self.currentPos[0] = 0 
		elif self.currentPos[0] >= self.rowsNb: # Outside from the bottom
			self.currentPos[0] = self.rowsNb-1 

		if self.currentPos[1] < 0: # Outside from the left 
			self.currentPos[1] = 0
		elif self.currentPos[1] >= self.colsNb: # Outisde from the right
			self.currentPos[1] = self.colsNb-1 

		#print(self.currentPos)

	def init_Q(self, actionsNb):
		Q = {}

		# Init keys
		states = []
		for i in range(self.rowsNb):
			for j in range(self.colsNb):
				states.append(str(i)+","+str(j))

		# For each key, give actions
		for state in states:
			Q[state] = [0 for _ in range(actionsNb)]

		return Q

	def choseAction(self, Q, eps, state):
		if random.random() < eps:
			return random.randint(0,7)
		else:
			return np.argmax(Q[state])

	def qLearning(self, episodes=5000, alpha=0.1, gamma=0.9, epsilon=0.2):
		# Initialize Q(s,a)
		Q = self.init_Q(8)

		# For the first graph plot
		episodesRewards = []
		# For the second graph plot
		stepsToGoalPerEpisode = []

		for i in range(episodes):
			grid = Windy_Gridworld() # Init s

			totalReward = 0
			stepsToGoal = 0

			while(grid.getCurrentPos() != grid.getGoal()):
				# Chose an action (direction) 
				currentState = str(grid.getCurrentPos()[0])+","+str(grid.getCurrentPos()[1])
				indexActionChosen = self.choseAction(Q, epsilon, currentState) 
				actionChosen = self.actionsIndex[indexActionChosen] 

				# Move to the direction chosen 
				grid.move(actionChosen) 

				# Compute reward
				reward = 10 if grid.getCurrentPos() == grid.getGoal() else -1
				totalReward += reward

				# Update Q 
				nextState = str(grid.getCurrentPos()[0])+","+str(grid.getCurrentPos()[1])
				Q[currentState][indexActionChosen] = Q[currentState][indexActionChosen] + alpha * (reward + gamma * max(Q[nextState]) - Q[currentState][indexActionChosen])

				stepsToGoal += 1 

			episodesRewards.append(totalReward)
			stepsToGoalPerEpisode.append(stepsToGoal)

		return Q, episodesRewards, stepsToGoalPerEpisode

	def displayArrows(self, Q):
		U, V = self.initArrows()
		X, Y = self.initXY()

		for i in range(self.rowsNb):
			for j in range(self.colsNb):
				currentState = str(i)+","+str(j)
				currentActions = Q.get(currentState)
				indexOfBestRewardAction = currentActions.index(max(currentActions))
				current_U, current_V = self.drawArrow(self.actionsIndex[indexOfBestRewardAction])
				if [i,j] == self.goal:
					current_U = 0
					current_V = 0
				U[i][j] = current_U
				V[i][j] = current_V
		
		plt.quiver(X,Y,U,V, scale=1,  units="xy", color='b', pivot='middle')

	def displayPath(self, Q, epsilon):
		grid = Windy_Gridworld()
		U, V = self.initArrows()
		X, Y = self.initXY()

		while(grid.getCurrentPos() != grid.getGoal()):
			# Chose action 
			currentState = str(grid.getCurrentPos()[0])+","+str(grid.getCurrentPos()[1])
			indexActionChosen = self.choseAction(Q, epsilon, currentState) 
			actionChosen = self.actionsIndex[indexActionChosen]

			# Draw the arrow correspoding to the action 
			arrow_U, arrow_V = self.drawArrow(actionChosen)
			x = grid.getCurrentPos()[0]
			y = grid.getCurrentPos()[1]
			U[x][y] = arrow_U
			V[x][y] = arrow_V

			# Move to the next action
			grid.move(actionChosen)

		plt.quiver(X,Y,U,V, scale=1,  units="xy", color='r', pivot='mid')
	
	def initXY(self):
		X = np.linspace(0,self.colsNb,self.colsNb)
		Y = np.linspace(self.rowsNb,0,self.rowsNb)
		return X,Y

	def initArrows(self):
		U = [[0 for i in range(self.colsNb)] for j in range(self.rowsNb)]
		V = [[0 for i in range(self.colsNb)] for j in range(self.rowsNb)]
		return U, V

	def drawArrow(self, direction):
		arrowSize = 0.5

		# Manage 8 directions 
		if direction == "W": 
			U = -arrowSize 
			V = 0 
		elif direction == "NW":
			U = -arrowSize 
			V = arrowSize
		elif direction == "N":
			U = 0 
			V = arrowSize 
		elif direction == "NE": 
			U = arrowSize
			V = arrowSize
		elif direction == "E":
			U = arrowSize 
			V = 0
		elif direction == "SE":
			U = arrowSize 
			V = -arrowSize 
		elif direction == "S":
			U = 0 
			V = -arrowSize 
		elif direction == "SW":
			U = -arrowSize 
			V = -arrowSize
		else: 
			U = 0 
			V = 0

		# If arrived to goal
		if self.currentPos == self.goal:
			U = 0
			V = 0

		return U,V

if __name__ == "__main__":
	myGrid = Windy_Gridworld()

	q, totalRewardPerEpisode, stepsToGoalPerEpisode = myGrid.qLearning(gamma=0.9, epsilon=0.2)

	myGrid.displayArrows(q)
	myGrid.displayPath(q, 0)

	plt.xticks(np.arange(0,myGrid.colsNb+1))
	plt.yticks(np.arange(0,myGrid.rowsNb+1))
	plt.grid(color='black', linestyle='dotted', linewidth=1)

	plt.title("Windy Gridworld execution")

	plt.show()

	# First graph plot 
	plt.plot(totalRewardPerEpisode)
	plt.title("Total collected reward per episode")
	plt.ylabel("Total reward")
	plt.xlabel("Episode number")
	plt.show()

	# Second graph plot
	plt.plot(stepsToGoalPerEpisode)
	plt.title("Number of steps to reach the goal")
	plt.ylabel("Steps")
	plt.xlabel("Episode number")
	plt.show()

