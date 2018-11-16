from __future__ import print_function
import argparse
from opti_node import opti_node
from arguments import get_args
import glob
import os
import pandas as pd
import numpy as np


def main():
	args = get_args()

	# File paths
	osm_nodes_file = '../devensData/devens_osm/osm_nodes.txt'
	osm_edges_file = '../devensData/devens_osm/osm_edges.txt'
	osm_nodes_gt_file = '../devensData/devens_osm/osm_nodes_truth.txt'
	osm_edges_gt_file = '../devensData/devens_osm/osm_edges_truth.txt'
	plot_dir = '../cache/'
	full_scan_file = '/home/sai/Downloads/truth.pkl'

	# Load in data
	osm_nodes = np.genfromtxt(osm_nodes_file, delimiter = ',')
	osm_edges = np.genfromtxt(osm_edges_file, delimiter = ',', dtype = int)
	osm_nodes_gt = np.genfromtxt(osm_nodes_gt_file, delimiter = ',')
	osm_edges_gt = np.genfromtxt(osm_edges_gt_file, delimiter = ',', dtype = int)
	full_scan = pd.read_pickle(full_scan_file)

	opti_node(args.variance, args.lamda, args.opti_method, args.osm_thresh, osm_nodes, osm_edges, osm_nodes_gt, osm_edges_gt, full_scan)

main()