# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 22:50:44 2018

@author: teddy
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

#%% Load from file
nodes = np.genfromtxt('osm_nodes.txt', delimiter=',')
edges = np.genfromtxt('osm_edges.txt', delimiter=',', dtype=int)

#%% Create a graph object
G = nx.Graph()
G.add_nodes_from(range(len(nodes)))
G.add_edges_from(edges)

#%% Plot
nx.draw(G, pos=nodes)
plt.show()

