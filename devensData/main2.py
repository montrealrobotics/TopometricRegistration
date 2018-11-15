import _pickle as pkl
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import shapely.geometry as geo
import numpy as np
import networkx as nx
import pandas as pd
from scipy.optimize import minimize
import copy

# File paths
osm_nodes_file = './devensData/devens_osm/osm_nodes.txt'
osm_edges_file = './devensData/devens_osm/osm_edges.txt'
osm_nodes_gt_file = './devensData/devens_osm/osm_nodes_truth.txt'
osm_edges_gt_file = './devensData/devens_osm/osm_edges_truth.txt'
plot_dir = './cache/'
full_scan_file = '/home/sai/Downloads/truth.pkl'

# Load in data
osm_nodes = np.genfromtxt(osm_nodes_file, delimiter = ',')
osm_edges = np.genfromtxt(osm_edges_file, delimiter = ',', dtype = int)
osm_nodes_gt = np.genfromtxt(osm_nodes_gt_file, delimiter = ',')
osm_edges_gt = np.genfromtxt(osm_edges_gt_file, delimiter = ',', dtype = int)
full_scan = pd.read_pickle(full_scan_file)


def transform_devens_2d(x):
	## This function takes in the following inputs: 
	# - the optimized parameters: x[0] (theta), x[1] (t_x), x[2] (t_y)
	# - Active OSM nodes (which were sampled from within 'osm_thresh' radius of scan position)
	# And transforms these active OSM nodes using the optimized parameters.
	# It returns the following:
	# - transformed OSM nodes
	osm_new = np.zeros((osm_nodes_active.shape[0], 2))
	for lol1 in range(osm_nodes_active.shape[0]):
		x_new = np.cos(x[0]) * osm_nodes_active[lol1,0] + np.sin(x[0]) * osm_nodes_active[lol1,1] + x[1]
		y_new = -np.sin(x[0]) * osm_nodes_active[lol1,0] + np.cos(x[0]) * osm_nodes_active[lol1,1] + x[2]

		osm_new[lol1,0] = x_new
		osm_new[lol1,1] = y_new
	return osm_new


def find_closest(x_new, y_new, scan_active):
	## This function takes in the following inputs:
	# - (x_new, y_new). These are the transformed OSM coordinates
	# - All the active scan points (scan points which are on the road)
	# And outputs the following:
	# - distance (in meters) of (x_new, y_new) to the closest scan point
	closest = 100000.0
	for lol2 in range(scan_active.shape[0]//100):
		dist = np.sqrt((x_new - scan_active[lol2*100,0])**2 + (y_new - scan_active[lol2*100,1])**2)
		if dist == 0:
			return dist
		if dist < closest:
			closest = dist

	return closest



def cost_func_2d(x):
	## This is the function we want to optimize.
	# The cost function has 2 parts
	my_cost1 = 0
	for lol3 in range(osm_nodes_active.shape[0]):
		dx = np.random.normal(0, 0)
		dy = np.random.normal(0, 0)
		x_noise = osm_nodes_active[lol3,0] + dx
		y_noise = osm_nodes_active[lol3,1] + dy
		x_new = np.cos(x[0]) * x_noise + np.sin(x[0]) * y_noise + x[1]
		y_new = -np.sin(x[0]) * x_noise + np.cos(x[0]) * y_noise + x[2]
		my_cost1 = my_cost1 + find_closest(x_new, y_new, scan_active)
	my_cost2 = 0
	for loll3 in range(osm_nodes_active.shape[0]):
		for loll4 in range(loll3+1, osm_nodes_active.shape[0]):
			my_cost2 = my_cost2 + np.sqrt((osm_nodes_active[loll3,0] - osm_nodes_active[loll4,0])**2 + (osm_nodes_active[loll3,1] - osm_nodes_active[loll4,1])**2)
#     print(my_cost1, mycost2)
	total_cost = my_cost1 - 0.01 * my_cost2
#     print(my_cost1, my_cost2, total_cost)
	return my_cost1



def find_error(osm1, osm2):
	## This function is not used in optimization. This is only a post-optimization check
	masks = np.zeros(osm2.shape[0])
	error = 0
	for lol4 in range(osm1.shape[0]):
		closest = 10000.0
		for lol5 in range(osm2.shape[0]):
			if masks[lol5] == 0 or masks[lol5] == 1:
				dist = np.sqrt((osm1[lol4,0] - osm2[lol5,0])**2 + (osm1[lol4,1] - osm2[lol5,1])**2)
				if dist == 0:
					error = error + dist
					masks[lol5] = 1
					break
				if dist < closest:
					closest = dist
					indi = lol5
		error = error + closest
		masks[indi] = 1
	return error


for scan_idx in range(100):
	scan = full_scan.iloc[scan_idx]['scan']
	road_mask = full_scan.iloc[scan_idx]['is_road_truth']
	road_mask = np.where(road_mask)

	# Extract OSM nodes that are within the distance threshold
	pos_x = full_scan.iloc[scan_idx]['x']
	pos_y = full_scan.iloc[scan_idx]['y']
	pos_theta = full_scan.iloc[scan_idx]['theta']
	
	osm_thresh = 150.
	osm_x, osm_y = [], []
	for eth in range(osm_nodes.shape[0]):
		if np.sqrt((osm_nodes[eth,0] - pos_x)**2 + (osm_nodes[eth,1] - pos_y)**2) < osm_thresh:
			osm_x.append(osm_nodes[eth,0])
			osm_y.append(osm_nodes[eth,1])
	osm_arr_x = np.asarray(osm_x)
	osm_arr_y = np.asarray(osm_y)
	osm_nodes_active = np.column_stack((osm_arr_x, osm_arr_y))
	
	osm_gt_x, osm_gt_y = [], []
	for eth in range(osm_nodes_gt.shape[0]):
		if np.sqrt((osm_nodes_gt[eth,0] - pos_x)**2 + (osm_nodes_gt[eth,1] - pos_y)**2) < osm_thresh:
			osm_gt_x.append(osm_nodes_gt[eth,0])
			osm_gt_y.append(osm_nodes_gt[eth,1])
	osm_arr_gt_x = np.asarray(osm_gt_x)
	osm_arr_gt_y = np.asarray(osm_gt_y)
	osm_nodes_gt_active = np.column_stack((osm_arr_gt_x, osm_arr_gt_y))
		 
	# Extract LiDAR scan points that correspond to road
	scan_active = scan[road_mask]
	scan_active = np.asarray([[item[0] for item in scan_active], [item[1] for item in scan_active]]).T
	scan_active_copy = copy.deepcopy(scan_active)
	for btc in range(scan_active.shape[0]):
		scan_active[btc,0] = scan_active_copy[btc,0] * np.cos(pos_theta) + scan_active_copy[btc,1] * np.sin(pos_theta) + pos_x
		scan_active[btc,1] = -scan_active_copy[btc,1] * np.sin(pos_theta) + scan_active_copy[btc,1] * np.cos(pos_theta) + pos_y
		
	# Do Optimization
	x0 = np.array([0.1, 0.8, 1.2])
	res = minimize(cost_func_2d, x0, method='powell', options={'xtol':1e-8, 'disp':True})
	#res_bounded = minimize(cost_func_2d, x0, method='trust-constr', options={'verbose':1}, bounds=bounds)
	#print(res.x)
	
	# Transform the OSM nodes based on optimized theta, x, y values
	osm_trans = transform_devens_2d(res.x)
	
	print(osm_trans.shape)
	#Plotting the results
	# fig = plt.figure()
	plt.scatter(osm_trans[:,0], osm_trans[:,1], color='k')
	# nx.draw(G, pos=osm_nodes2)
	plt.show()
	# plt.pause(1)
	plt.savefig('./devensData/osm_trans'+str(scan_idx)+'.png')
	plt.gcf().clear()
	plt.scatter(osm_nodes_gt_active[:,0], osm_nodes_gt_active[:,1])
	plt.show()
	plt.savefig('./devensData/osm_gt'+str(scan_idx)+'.png')
	plt.gcf().clear()
	plt.scatter(scan_active[:,0], scan_active[:,1])
	plt.show()
	plt.savefig('./devensData/scan'+str(scan_idx)+'.png')
	plt.gcf().clear()
	plt.scatter(osm_nodes_gt_active[:,0], osm_nodes_gt_active[:,1])
	plt.scatter(osm_nodes_active[:,0], osm_nodes_active[:,1], color='k')
	plt.scatter(osm_trans[:,0], osm_trans[:,1])
	plt.scatter(scan_active[:,0], scan_active[:,1])
	plt.savefig('./devensData/combined'+str(scan_idx)+'.png')
	plt.gcf().clear()
	# ori_error = find_error(osm_nodes_active, osm_nodes_gt_active)
	# opti_error = find_error(osm_trans, osm_nodes_gt_active)
	# print(pos_x, pos_y)
	# print(osm_nodes_active)
	# print(osm_trans)
	# print(osm_nodes_gt_active)
	# print(ori_error, opti_error)