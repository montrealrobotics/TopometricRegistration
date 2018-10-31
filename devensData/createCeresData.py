import _pickle as pkl
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# File paths
devens_map_file = '/home/jatavalk/code/TopometricRegistration/devensData/devens_map/devens_map_poly.pkl'
osm_nodes_file = '/home/jatavalk/code/TopometricRegistration/devensData/devens_osm/osm_nodes.txt'
osm_edges_file = '/home/jatavalk/code/TopometricRegistration/devensData/devens_osm/osm_edges.txt'
# cloud_file = '/home/jatavalk/code/TopometricRegistration/devensData/pointclouds/scans_in_utm_small.pkl'
cloud_file = '/home/jatavalk/datasets/devens/scans_in_utm_annotated.pkl'
plot_dir = '/home/jatavalk/code/TopometricRegistration/cache/'


# Load in data
devens_map = pkl.load(open(devens_map_file, 'rb'), encoding = 'bytes')
osm_nodes = np.genfromtxt(osm_nodes_file, delimiter = ',')
osm_edges = np.genfromtxt(osm_edges_file, delimiter = ',', dtype = int)
cloud = pd.read_pickle(cloud_file)


# # Get road points
# interior_pts = None
# i = 0
# for obj in devens_map.interiors:
# 	if i >= 4:
# 		if interior_pts is None:
# 			interior_pts = np.concatenate([np.expand_dims(obj.xy[0], axis=1), np.expand_dims(obj.xy[1], axis=1)], axis=1)	
# 		else:
# 			interior_pts = np.concatenate([interior_pts, np.concatenate([np.expand_dims(obj.xy[0], axis = 1), np.expand_dims(obj.xy[1], axis = 1)], \
# 				axis = 1)], axis = 0)
# 		# print(interior_pts.shape)
# 		# plt.plot(obj.xy[0], obj.xy[1], 'k')
# 	i += 1

# # plt.scatter(interior_pts[:,0], interior_pts[:,1])
# # plt.savefig(plot_dir + 'tmp.png')


# # # Compute an approximate bounding rectangle of the road points
# # x_min = np.min(interior_pts[:,0])
# # x_max = np.max(interior_pts[:,0])
# # y_min = np.min(interior_pts[:,1])
# # y_max = np.max(interior_pts[:,1])

# # # Get osm nodes (that are within a 10 meter radius of the approx bounding rectangle)
# # osm_thresh = 10.

# # osm_indices = []
# # for i in range(osm_nodes.shape[0]):
# # 	if x_min < osm_nodes[i,0] < x_max and y_min < osm_nodes[i,1] < y_max:
# # 		osm_indices.append(i)

# # Compute the centroid of the road points
# x_cent = np.mean(interior_pts[:,0])
# y_cent = np.mean(interior_pts[:,1])

# # Get OSM nodes that are within a distance threshold from the centroid
# osm_thresh = 10000.
# osm_indices = []
# for i in range(osm_nodes.shape[0]):
# 	if np.sqrt((osm_nodes[i,0]-x_cent)**2 + (osm_nodes[i,1]-y_cent)**2) <= osm_thresh:
# 		osm_indices.append(i)
# print(len(osm_indices))
# plt.scatter([osm_nodes[i,0] for i in osm_indices], [osm_nodes[i,1] for i in osm_indices])
# plt.savefig(plot_dir + 'tmp.png')


# scan = cloud.iloc[0]['scan_utm']
# print(scan['x'].shape, scan['y'].shape)
