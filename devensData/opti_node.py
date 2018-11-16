import _pickle as pkl
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import shapely.geometry as geo
import numpy as np
import networkx as nx
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import copy
import time
from datetime import datetime

class opti_node(object):
	def __init__(self, variance, lamda, opti_method, osm_thresh, osm_nodes, osm_edges, osm_nodes_gt, osm_edges_gt, full_scan):

		self.lamda = lamda
		self.opti_method = opti_method
		self.osm_thresh = osm_thresh
		self.variance = variance
		self.osm_nodes = osm_nodes
		self.osm_edges = osm_edges
		self.osm_nodes_gt = osm_nodes_gt
		self.osm_edges_gt = osm_edges_gt
		self.full_scan = full_scan
		
		self.scan_active = np.zeros((50000, 2))
		self.osm_nodes_active = np.zeros((50,2))
		# current_time = datetime.now().strftime('%b%d_%H-%M-%S')
		self.start_time = time.time()

		self.just_do_it()

	def transform_devens_2d(self, x):
		## This function takes in the following inputs: 
		# - the optimized parameters: x[0] (theta), x[1] (t_x), x[2] (t_y)
		# - Active OSM nodes (which were sampled from within 'osm_thresh' radius of scan position)
		# And transforms these active OSM nodes using the optimized parameters.
		# It returns the following:
		# - transformed OSM nodes
		rotation_mat = [[np.cos(x[0]), np.sin(x[0])],
						 [-np.sin(x[0]), np.cos(x[0])]]
		trans_mat = [[x[1], x[2]]]
		osm_new = np.matmul(self.osm_nodes_active, rotation_mat) + trans_mat
		return osm_new


	def cost_func_2d(self, x):
		## This is the function we want to optimize.
		# The cost function has 2 parts
		#
		# - cost1 is the summation of distance from every transformed OSM node to it's corresponding closest scan point.
		# - Basically, "What should be the transformation parameters such that the distance between transformed OSM nodes and 
		# - their closest scan points is reduced? "
		#
		# - cost2 is the summation of distances between the OSM nodes. So, if there are n OSM nodes, we will sum over C(n,2) such distances
		# - This is to ensure that all OSM nodes doesn't collapse to a signle scan point.
		# - We want this cost2 to be high. Hence, the negative sign
		# - The weightage given to cost2 can be changed with hyper parameter \lamda 
		
		rotation_mat = [[np.cos(x[0]), np.sin(x[0])],
						 [-np.sin(x[0]), np.cos(x[0])]]
		trans_mat = [[x[1], x[2]]]
		osm_new = np.matmul(self.osm_nodes_active, rotation_mat) + trans_mat
		
		cost1 = np.sum((cdist(osm_new, self.scan_active, 'euclidean')).min(axis=1)) 
		cost2 = 0.5 * np.sum((cdist(osm_new, osm_new, 'euclidean')))
		
		total_cost = cost1 - self.lamda * cost2
		return total_cost

	def find_error(self, osm1, osm2):
		## This function is not used in optimization. This is only a post-optimization check
		# Typically, osm2 is the groundtruth OSM nodes
		# - We want to measure the shortest distance between optimized or unoptimized OSM nodes (depending on what we send as input)
		# vs ground truth OSM nodes. 
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


	def just_do_it(self):
		for scan_idx in range(100): # Just taking in the first 100 scans
			scan = self.full_scan.iloc[scan_idx]['scan_utm']
			road_mask = self.full_scan.iloc[scan_idx]['is_road_truth']
			road_mask = np.where(road_mask)

			# Extract OSM nodes that are within the distance threshold
			pos_x = self.full_scan.iloc[scan_idx]['x']
			pos_y = self.full_scan.iloc[scan_idx]['y']
			pos_theta = self.full_scan.iloc[scan_idx]['theta']
			
			osm_x, osm_y = [], []
			for eth in range(self.osm_nodes.shape[0]):
				if np.sqrt((self.osm_nodes[eth,0] - pos_x)**2 + (self.osm_nodes[eth,1] - pos_y)**2) < self.osm_thresh:
					osm_x.append(self.osm_nodes[eth,0])
					osm_y.append(self.osm_nodes[eth,1])
			osm_arr_x = np.asarray(osm_x)
			osm_arr_y = np.asarray(osm_y)
			osm_nodes_active = np.column_stack((osm_arr_x, osm_arr_y))
			rand_noise = np.random.normal(0, self.variance, (osm_nodes_active.shape[0], 2))
			osm_nodes_active = osm_nodes_active + rand_noise

			osm_gt_x, osm_gt_y = [], []
			for eth in range(self.osm_nodes_gt.shape[0]):
				if np.sqrt((self.osm_nodes_gt[eth,0] - pos_x)**2 + (self.osm_nodes_gt[eth,1] - pos_y)**2) < self.osm_thresh:
					osm_gt_x.append(self.osm_nodes_gt[eth,0])
					osm_gt_y.append(self.osm_nodes_gt[eth,1])
			osm_arr_gt_x = np.asarray(osm_gt_x)
			osm_arr_gt_y = np.asarray(osm_gt_y)
			osm_nodes_gt_active = np.column_stack((osm_arr_gt_x, osm_arr_gt_y))
				 
			
			# Extract LiDAR scan points that correspond to road
			scan_active = scan[road_mask]
			scan_active = np.asarray([[item[0] for item in scan_active], [item[1] for item in scan_active]]).T
			# scan_active_copy = copy.deepcopy(scan_active)
			# rotation_mat = [[np.cos(pos_theta), np.sin(pos_theta)],
			# 			 [-np.sin(pos_theta), np.cos(pos_theta)]]
			# trans_mat = [[pos_x, pos_y]]
			# scan_active = np.matmul(scan_active_copy, rotation_mat) + trans_mat 

			self.osm_nodes_active = osm_nodes_active
			self.scan_active = scan_active

			# Do Optimization
			x0 = np.array([0.1, 0.8, 1.2])
			if self.opti_method == 'powell':
				res = minimize(self.cost_func_2d, x0, method='powell', options={'xtol':1e-8, 'disp':True})
			elif self.opti_method == 'nelder-mead':
				res = minimize(self.cost_func_2d, x0, method='nelder-mead', options={'xtol':1e-8, 'disp':True})
			

			# Transform the OSM nodes based on optimized theta, x, y values
			osm_trans = self.transform_devens_2d(res.x)
			
			#Plotting the results
			plt.scatter(osm_trans[:,0], osm_trans[:,1], color='k')
			plt.show()
			plt.savefig('../devensData/osm_trans'+str(scan_idx)+'.png')
			plt.gcf().clear()
			plt.scatter(osm_nodes_gt_active[:,0], osm_nodes_gt_active[:,1])
			plt.show()
			plt.savefig('../devensData/osm_gt'+str(scan_idx)+'.png')
			plt.gcf().clear()
			plt.scatter(scan_active[:,0], scan_active[:,1])
			plt.show()
			plt.savefig('../devensData/scan'+str(scan_idx)+'.png')
			plt.gcf().clear()
			plt.scatter(osm_nodes_gt_active[:,0], osm_nodes_gt_active[:,1])
			plt.scatter(osm_nodes_active[:,0], osm_nodes_active[:,1], color='k')
			plt.scatter(osm_trans[:,0], osm_trans[:,1])
			plt.scatter(scan_active[:,0], scan_active[:,1])
			plt.savefig('../devensData/combined'+str(scan_idx)+'.png')
			plt.gcf().clear()
			ori_error = self.find_error(osm_nodes_active, osm_nodes_gt_active)
			opti_error = self.find_error(osm_trans, osm_nodes_gt_active)
			print(ori_error, opti_error)
			print(time.time() - self.start_time)
		print(time.time() - self.start_time)