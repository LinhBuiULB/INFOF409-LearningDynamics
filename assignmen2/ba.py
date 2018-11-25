import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from scipy.stats import norm

def init_4nodes_graph():
   G = nx.Graph()

   nodes = [0,1,2,3]
   edges = [(i,j) for i in range(len(nodes)) for j in range(i)]

   G.add_nodes_from(nodes)
   G.add_edges_from(edges)

   return G 

def barabasi_albert(G):
   stop = 10
   probaLink = 0

   for i in range(3,stop):
      G.add_node(i)
      nbLinks = 0 

      while(nbLinks < 4):
         for j in range(len(G)):
            probaLink = computeProbaLinkTo_i(G,j)
            if random.random() < probaLink:
               G.add_edge(i, j)
               nbLinks += 1
               if(nbLinks == 4): break 

def sumAllNodesDegrees(G):
   allNodes = list(G.nodes)
   tuplesNodesDegrees = list(G.degree(allNodes))
   sumAllNodesDegrees = 0

   for i in range(len(tuplesNodesDegrees)):
      sumAllNodesDegrees += tuplesNodesDegrees[i][1]

   return sumAllNodesDegrees

def computeProbaLinkTo_i(G, node_i):
   prob_linkTo_i = G.degree(node_i) / sumAllNodesDegrees(G)
   return prob_linkTo_i

def plotDegreeDistribution(G):
   degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
   # print "Degree sequence", degree_sequence
   degreeCount = collections.Counter(degree_sequence)
   deg, cnt = zip(*degreeCount.items())
   fig, ax = plt.subplots()
   hist, bins, _ = plt.hist(degree_sequence, bins=25, density=True, alpha=0.6, color='g')

   """ 
   # LOG SCALE 
   logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
   plt.subplot(212)
   plt.hist(degree_sequence, bins=logbins)
   plt.xscale('log')
   """

   plt.title("Degree Histogram")
   plt.ylabel("Count")
   plt.xlabel("Degree")
   ax.set_xticks([d + 0.4 for d in deg])
   ax.set_xticklabels(deg)

   plt.show()

def initGraph():
   G=nx.Graph()
   return G

def printGraph(G):
   nx.draw(G,with_labels=True)
   plt.show() # display

def main():
   G = init_4nodes_graph()

   barabasi_albert(G)

   print(G.nodes())
   print(G.edges())

   plotDegreeDistribution(G)
   printGraph(G)

if __name__ == "__main__":
   main()