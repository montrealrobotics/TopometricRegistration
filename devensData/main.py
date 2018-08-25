import cPickle as pkl
import matplotlib.pyplot as plt
import shapely.geometry as geo
import numpy as np
import networkx as nx
import pandas as pd

#nodes = np.genfromtxt('devens_osm/osm_nodes.txt', delimiter=',')
nodes = pd.read_csv('devens_osm/osm_nodes.txt', header=None)
edges = np.genfromtxt('devens_osm/osm_edges.txt', delimiter=',', dtype=int)
devens = pkl.load(open('devens_map/devens_map_poly.pkl', 'r'))
df = pkl.load(open('pointclouds/scans_in_utm_small.pkl', 'r'))

G = nx.Graph()
#G.add_nodes_from(range(len(nodes)))
i = 0
for row in nodes.iterrows():
    G.add_node(i, x=row[1][0], y=row[1][1])
    i = i+1
xs = nx.get_node_attributes(G, 'x')
print(xs[0])
G.add_edges_from(edges)
scan = df.iloc[0]['scan_utm']


plot_coords = lambda obj: plt.plot(obj.xy[0],obj.xy[1], 'k')
plot_coords(devens.exterior)
[plot_coords(x) for x in devens.interiors]
print "Is the origin on the road? {}".format(devens.contains(geo.Point([0,0])))
plt.plot(0,0,'gx')
plt.show()

nx.draw(G, pos=nodes)
plt.show()
