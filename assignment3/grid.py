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

		print(self.currentPos)

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

		for i in range(episodes):
			grid = Windy_Gridworld() # Init s
			while(grid.getCurrentPos() != grid.getGoal()):
				# Chose an action (direction) 
				currentState = str(grid.getCurrentPos()[0])+","+str(grid.getCurrentPos()[1])
				indexActionChosen = self.choseAction(Q, epsilon, currentState) 
				actionChosen = self.actionsIndex[indexActionChosen] 

				# Move to the direction chosen 
				grid.move(actionChosen) 

				# Compute reward
				reward = 10 if grid.getCurrentPos() == grid.getGoal() else -1

				# Update Q 
				nextState = str(grid.getCurrentPos()[0])+","+str(grid.getCurrentPos()[1])
				Q[currentState][indexActionChosen] = Q[currentState][indexActionChosen] + alpha * (reward + gamma * max(Q[nextState]) - Q[currentState][indexActionChosen])

		return Q


if __name__ == "__main__":
	myGrid = Windy_Gridworld()

	# Test windy move 
	myGrid.setCurrentPos([4,5])
	print("Starting from [4,5], going E, arriving at :")
	myGrid.move("E")

	# Test outside move 
	myGrid.setCurrentPos([0,11])
	print("Starting from [0,11], going NW, arriving at:")
	myGrid.move("NW")

	q = myGrid.qLearning()
	print(q)

