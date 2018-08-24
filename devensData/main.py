import cPickle as pkl
import matplotlib.pyplot as plt
import shapely.geometry as geo
import numpy as np
import networkx as nx

nodes = np.genfromtxt('devens_osm/osm_nodes.txt', delimiter=',')
edges = np.genfromtxt('devens_osm/osm_edges.txt', delimiter=',', dtype=int)
devens = pkl.load(open('devens_map/devens_map_poly.pkl', 'r'))
df = pkl.load(open('pointclouds/scans_in_utm_small.pkl', 'r'))

G = nx.Graph()
G.add_nodes_from(range(len(nodes)))
G.add_edges_from(edges)
scan = df.iloc[0]['scan_utm']

