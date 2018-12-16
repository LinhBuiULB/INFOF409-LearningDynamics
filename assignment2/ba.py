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
   xx = np.linspace(3, 400, 1000)
   yy = exponential_func(xx, *popt)
   return xx, yy

def plotDegreeDistribution(G):
   degree_sequence, deg, cnt = getDegreesCount(G)
   print(degree_sequence,deg,cnt)
   degree_sequence = np.array(degree_sequence)

   # Max Likelikhood 
   results = powerlaw.Fit(degree_sequence)
   fig2 = results.plot_pdf(label="Empirical distribution")
   results.power_law.plot_pdf(color="b", linestyle="--", ax=fig2, label="Fitted distribution")
   print("Alpha = ", results.power_law.alpha)
   print("Sigma = ", results.power_law.sigma)
   R, p = results.distribution_compare('power_law', 'exponential', normalized_ratio=True)
   print("R =",R , "p =",p)
   #plt.plot(x,y)

   # Plot exponential fit 
   #xx,yy = exponentialFit(deg,cnt)
   #plt.plot(xx, yy,'-', color="red", label="Exponential distribution")

   # Plot degree distribution 
   #plt.bar(deg, cnt, align='center', width=0.80, color='b', label="Degree distribution")
   #plt.plot(deg,cnt,'o')
   plt.legend(loc='upper right')
   # LOG SCALE 
   #plt.xscale('log')
   #plt.yscale('log')

   title = "Fitting with Maximum Likelihood method: alpha = %.2f,  sigma = %.2f" % (results.power_law.alpha, results.power_law.sigma)
   plt.title(title)
   plt.ylabel("p(degree)")
   plt.xlabel("Degree Frequency")

   plt.show()

def initGraph():
   G=nx.Graph()
   return G

def printGraph(G):
   nx.draw(G,with_labels=True)
   plt.show() # display

def main():
   G = init_4nodes_graph()

   barabasi_albert(G, 10000)

   #print(G.nodes())
   #print(G.edges())

   plotDegreeDistribution(G)
   #printGraph(G)

if __name__ == "__main__":
   main()