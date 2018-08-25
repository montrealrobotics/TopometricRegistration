import cPickle as pkl
import matplotlib.pyplot as plt
import shapely.geometry as geo
import numpy as np
import networkx as nx
import pandas as pd
from scipy.optimize import minimize



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
ys = nx.get_node_attributes(G, 'y')
#print(xs[0])
G.add_edges_from(edges)
scan = df.iloc[0]['scan_utm']
#print(len(devens.exterior))
#print(devens.area)
#print(devens.bounds)
#print(len(list(devens.exterior.coords)))
#print(devens.exterior.coords.size())
#print(devens.interiors.coords) #not existant
devens_arr = np.asarray(list(devens.exterior.coords))
#print(devens_arr[100,1])


def find_closest(x_new, y_new):
    closest = 100000.0
    for j in range(len(xs)):
        dist = np.sqrt((x_new - xs[j])**2 + (y_new - ys[j])**2)
        if dist < closest:
            closest = dist

    return closest

def cost_func(x):
    my_cost = 0
    for i in range(len(list(devens.exterior.coords))):
        x_dash = x[0] * devens_arr[i,0] + x[1] * devens_arr[i,1] + x[4]
        y_dash = x[2] * devens_arr[i,0] + x[1] * devens_arr[i,1] + x[5]
        z_dash = 1 + x[6]
        x_new = x_dash / z_dash
        y_new = y_dash / z_dash

        my_cost = my_cost + find_closest(x_new, y_new)
        return my_cost

#print(devens.exterior.coords[1,0])
x0 = np.array([0.5, 0.1, 0.1, 0.5, 1.0, 0.8, 1.2])
res = minimize(cost_func, x0, method='nelder-mead', options={'xtol':1e-8, 'disp':True})
print(res.x)


#plot_coords = lambda obj: plt.plot(obj.xy[0],obj.xy[1], 'k')
#plot_coords(devens.exterior)
#[plot_coords(x) for x in devens.interiors]
#print "Is the origin on the road? {}".format(devens.contains(geo.Point([0,0])))
#plt.plot(0,0,'gx')
#plt.show()

#nx.draw(G, pos=nodes)
#plt.show()
