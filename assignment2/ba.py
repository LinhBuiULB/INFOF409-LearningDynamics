import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from scipy.stats import norm

TOTAL_DEGREE = 12

def init_4nodes_graph():
   G = nx.Graph()

   nodes = [0,1,2,3]
   edges = [(i,j) for i in range(len(nodes)) for j in range(i)]

   G.add_nodes_from(nodes)
   G.add_edges_from(edges)

   return G 

def barabasi_albert(G):
   stop = 100
   probaList = []
   global TOTAL_DEGREE

   for i in range(4,stop):
      G.add_node(i)
      print(i)
      for j in range(0,i):
         probaList.append(computeProbaLinkTo_i(G,j))
      choseNodes = np.random.choice(i, 4, replace=False, p=probaList)
      for node in choseNodes:
          G.add_edge(i, node)
      TOTAL_DEGREE += 8 
      probaList = []

def computeProbaLinkTo_i(G, node_i):
   prob_linkTo_i = G.degree(node_i) / TOTAL_DEGREE
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
   plt.bar(deg, cnt, align='center', width=0.80, color='b')

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

   #print(G.nodes())
   #print(G.edges())

   plotDegreeDistribution(G)
   #printGraph(G)

if __name__ == "__main__":
   main()