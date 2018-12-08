import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from scipy.stats import norm, binned_statistic
from scipy.optimize import curve_fit
import powerlaw 

TOTAL_DEGREE = 12

def init_4nodes_graph():
   G = nx.Graph()

   nodes = [0,1,2,3]
   edges = [(i,j) for i in range(len(nodes)) for j in range(i)]

   G.add_nodes_from(nodes)
   G.add_edges_from(edges)

   return G 

def barabasi_albert(G, stop):
   probaList = []
   global TOTAL_DEGREE

   for i in range(4,stop):
      G.add_node(i)
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

def getDegreesCount(G):
   degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
   # print "Degree sequence", degree_sequence
   degreeCount = collections.Counter(degree_sequence)
   deg, cnt = zip(*degreeCount.items())
   return degree_sequence, deg, cnt

def exponential_func(x, a, b, c):
   return a*np.exp(-b*x)+c

def exponentialFit(deg, cnt):
   popt, pcov = curve_fit(exponential_func, deg, cnt, p0=(1, 1e-6, 1))
   xx = np.linspace(3, 100, 1000)
   yy = exponential_func(xx, *popt)
   return xx, yy

def plotDegreeDistribution(G):
   degree_sequence, deg, cnt = getDegreesCount(G)
   print(degree_sequence,deg,cnt)
   degree_sequence = np.array(degree_sequence)
   
   plt.title("Degree Histogram")
   plt.ylabel("Count")
   plt.xlabel("Degree")

   # Max Likelikhood 
   results = powerlaw.Fit(degree_sequence)
   results.plot_pdf()
   #R, p = results.distribution_compare('power_law', 'lognormal')
   #print("R,p",R,p)
   #plt.plot(x,y)

   # Plot exponential fit 
   #xx,yy = exponentialFit(deg,cnt)
   #plt.plot(xx, yy,'-', color="red")

   # Plot degree distribution 
   #plt.bar(deg, cnt, align='center', width=0.80, color='b')
   #plt.plot(deg,cnt,'o')

   # LOG SCALE 
   plt.xscale('log')
   plt.yscale('log')

   plt.show()

def initGraph():
   G=nx.Graph()
   return G

def printGraph(G):
   nx.draw(G,with_labels=True)
   plt.show() # display

def main():
   G = init_4nodes_graph()

   barabasi_albert(G, 1000)

   #print(G.nodes())
   #print(G.edges())

   plotDegreeDistribution(G)
   #printGraph(G)

if __name__ == "__main__":
   main()