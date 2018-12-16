import numpy as np 
import matplotlib.pyplot as plt
import random 


ACTIONS = [(2.4, 0.9), (1.3, 0.6), (1, 0.4), (2.2, 2)]
ACTIONS_NUMBER = 4 
ALPHA = 0.1 
GAMMA = 0.9
ITERATIONS = 1000

EPSILON = [0,0.1,0.2]
TAU = [1,0.1]

rewards = {1:[], 2:[], 3:[], 4:[]}
Q = [0,0,0,0]

def computeRewardDistribution(mu, sigma):
	return np.random.normal(mu, sigma, 1000)

def putNewRewards():
	for i in range(len(ACTIONS)):
		rewards[i+1] = computeRewardDistribution(ACTIONS[i][0],ACTIONS[i][1])
	return rewards

def softmaxProbability(action, tau): 
	num = np.exp(Q[action]/tau)
	denom = 0

	for i in range(ACTIONS_NUMBER): 
		denom += np.exp(Q[i]/tau)

	return num/denom

def chooseArm(strategy, eps, tau):
	if strategy == "random": 
		return random.randint(1,ACTIONS_NUMBER)

	elif strategy == "greedy":
		if random.random() < eps:
			return random.randint(1,ACTIONS_NUMBER)
		else:
			return np.argmax(Q)+1

	elif strategy == "softmax":
		while True: 
			for i in range(ACTIONS_NUMBER):
				if random.random() < softmaxProbability(i, tau):
					return i+1 

def bandit(strategy, eps=0, tau=0):
	actionsDone = []
	actionsRewards = []
	Q_0 = [] # Arm 1 
	Q_1 = [] # Arm 2 
	Q_2 = [] # Arm 3 
	Q_3 = [] # Arm 4 

	for t in range(ITERATIONS):
		actionPicked = chooseArm(strategy, eps, tau)
		actionsDone.append(actionPicked)
		actionsRewards.append(rewards[actionPicked][t])

		sumOfReward = 0
		hasBeenTaken = 0 
		for i in range(t):
			if actionsDone[i] == actionPicked: 
				sumOfReward += actionsRewards[i]
				hasBeenTaken += 1

		if(hasBeenTaken != 0): 
			Q[actionPicked-1] = sumOfReward / hasBeenTaken
		else: 
			Q[actionPicked-1] = 0

		Q_0.append(Q[0])
		Q_1.append(Q[1])
		Q_2.append(Q[2])
		Q_3.append(Q[3])

	print(Q)
	return Q, sumOfReward, Q_0, Q_1, Q_2, Q_3 

def main():
	putNewRewards()

	algosList = []
	avgRewards = []

	# Random 
	print("Random :")
	randomQ, rewardQ, Q_0, Q_1, Q_2, Q_3 = bandit(strategy="random")
	# For the first plot 
	avgRewards.append(rewardQ/ITERATIONS)
	algosList.append("R")

	# For the second plot 
	plt.plot(Q_1, label="Random")  #######

	print("\n")

	# Epsilon Greedy 
	for eps in EPSILON:
		print("eps =",eps)
		epsQ, epsReward, Q_0, Q_1, Q_2, Q_3 = bandit(strategy="greedy",eps=eps)
		avgRewards.append(epsReward/ITERATIONS)
		algosList.append("eps={}".format(eps))

		# For the second plot 
		plt.plot(Q_1, label="Greedy - epsilon = {}".format(eps)) #######

	print("\n")

	# Softmax 
	for val in TAU:
		print("tau =",val)
		tauQ, tauReward, Q_0, Q_1, Q_2, Q_3 = bandit(strategy="softmax",tau=val)
		avgRewards.append(tauReward/ITERATIONS)
		algosList.append("Softmax - tau = {}".format(val))

		# For the second plot 
		plt.plot(Q_1, label="tau={}".format(val)) #######

	print("\n")

	# First plot 
	#plt.plot(algosList, avgRewards)

	plt.hlines(y=1.3, xmin=0, xmax=1000, linewidth=3, color='black', linestyles="dotted", label="Theoretical") # Theoretical - Arm 1: 2.4 , Arm2 : 1.3,  Arm3 : 1, Arm4 : 2.2
	plt.title("ARM 2 - Values of Q(a) over time")
	plt.legend()
	plt.show()

if __name__ == "__main__":
	main()