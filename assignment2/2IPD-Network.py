import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from scipy.stats import norm
import er 
import ba

T = [1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]                    # Temptation to defect
R=1                  # Reward for mutual cooperation
P=0                   # Punishment for mutual defection
S=-0.1                    # Suckers payoff for unilateral cooperation
rounds=1000              # Number of rounds

actions = {'D': 0, 'C': 1} 

def IPD(player1, player2, number_t): 

  payoffs = np.array([
    [P, T[number_t]],
    [S, R]
  ])

  return payoffs[actions[player1]][actions[player2]]

def randomCoopOrDefect(G):
	"""
	Give random C or D strategy to each node of the graph
	"""
	nodeStrategy = {}
	for i in range(len(G)):
		nodeStrategy[i] = random.choice(["C","D"])
	nx.set_node_attributes(G, nodeStrategy, 'strategy')

def maxDegree(G,node1,node2):
  return max(G.degree(node1),G.degree(node2))

def getDmax(number_t):
  return max(T[number_t],R) - min(S,P)

def adaptStrategy(G, i, number_t, payoffs_nodes):

  if(len(list(G.neighbors(i))) != 0):

    j = random.choice(list(G.neighbors(i)))

    Pij = (payoffs_nodes[j]-payoffs_nodes[i]) / (maxDegree(G,i,j) * getDmax(number_t))
    
    rand = random.random()
    if(rand < Pij):
      G.nodes[i]['strategy'] = G.nodes[j]['strategy']


def playIPDwithNeighbors(G,node, number_t):
	"""
	Get the total payoff of a node playing IPD with all his neighbors
	"""
	totalPayoff = 0 
	for neigh in list(G.neighbors(node)):
		res = IPD(G.nodes[node]['strategy'], G.nodes[neigh]['strategy'], number_t)
		totalPayoff += res
	return totalPayoff

def IPDonNetwork(G,number_t):

  cooperationLevel = 0
  cooperationLevelList = []

  payoffs_nodes = []
  for i in range(len(G)):
    payoffs_nodes.append(0)

  # random C or D for each node 
  randomCoopOrDefect(G)

  # rest of round : adapt 
  for r in range(rounds):
    cooperationLevel = 0
    for i in range(len(G)): 
      payoff_current = playIPDwithNeighbors(G,i,number_t)
      payoffs_nodes[i] = payoff_current 
      adaptStrategy(G,i, number_t, payoffs_nodes)
      if(G.nodes[i]['strategy'] == 'C'): cooperationLevel += 1

    cooperationLevelList.append(cooperationLevel/(len(G)))
    print(r)

  return cooperationLevelList

def plotCooperationLevel(cooperationLevelList, number_t):
  legendLabel = "T = {}".format(T[number_t]) 
  plt.plot(cooperationLevelList, label=legendLabel)
  plt.legend(loc="upper left")


def main():

  network = "BA"

  if(network == "ER"): 
    n = 10000
    K = 20000

    G = er.initGraph()

    nodesG = [i for i in range(n)]
    edgesG = er.erdosRenyi(n,K)

    G.add_nodes_from(nodesG)
    G.add_edges_from(edgesG)

  elif(network == "BA"): 
    stop = 10000
    G = ba.init_4nodes_graph()
    ba.barabasi_albert(G, stop)


  cooperations = []

  for i in range(len(T)):
    coop = IPDonNetwork(G, i)
    cooperations.append(coop)
    plotCooperationLevel(coop, i)

  plt.show()

if __name__ == "__main__":
   main()