import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from scipy.stats import norm


def erdosRenyi(n,K):
	edges = []
	stop = 0

	for i in range(K): 
		firstNode = random.randint(0,n)
		secondNode = random.randint(0,n)
		newEdge = (firstNode,secondNode)
		if newEdge not in edges and (secondNode,firstNode) not in edges:
			edges.append(newEdge)

	return edges 


def initGraph():
	G=nx.Graph()
	return G

def printGraph(nodes, edges, G):
	G.add_nodes_from(nodes)
	G.add_edges_from(edges)

	nx.draw(G)
	plt.show() # display

def getDegreesCount(G):
	degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
	# print "Degree sequence", degree_sequence
	degreeCount = collections.Counter(degree_sequence)
	deg, cnt = zip(*degreeCount.items())
	return degree_sequence, deg, cnt

def fitNormal(degree_sequence):
	mu, std = norm.fit(degree_sequence)
	return mu, std

def plotDegreeDistribution(G):

	degree_sequence, deg, cnt = getDegreesCount(G)

	# Fit a normal distribution to the data:
	mu, std = fitNormal(degree_sequence)
	print("Mean",np.mean(degree_sequence))
	print("Std",np.std(degree_sequence))

	# Plot the histogram.
	plt.hist(degree_sequence, bins=25, density=True, alpha=0.6, color='blue', align='mid')

	# Plot the PDF.
	xmin, xmax = plt.xlim()
	x = np.linspace(xmin, xmax, 100)
	p = norm.pdf(x, mu, std)

	plt.plot(x, p, 'k', linewidth=2)
	title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
	plt.title(title)

	plt.show()

def main():
	n = 10000
	K = 20000

	G = initGraph()

	nodesG = [i for i in range(n)]
	edgesG = erdosRenyi(n,K)

	#print("Nodes :", nodesG)
	#print("Edges :", edgesG)

	G.add_nodes_from(nodesG)
	G.add_edges_from(edgesG)

	plotDegreeDistribution(G)

if __name__ == "__main__":
	main()