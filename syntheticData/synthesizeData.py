"""
Python script for synthesizing data for synthetic OSM and local sensor frame views
"""

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

	"""
	Generate a two-way intersection OSM graph
	"""

	# Number of nodes in the main branch, in the left turn, and in the right brach
	# Think of the intersection as a 'Y', where the vertical line in the bottom-half of the Y
	# represents the main branch, and the slanted lines in the top-half of the Y represent 
	# the left and right branches.
	# nodes = [10, 15, 20]
	nodes = [10, 5, 5]

	# Generate keys for the dictionary that's going to hold the graph
	keys = ['M'+str(i).zfill(2) for i in range(nodes[0])] + \
	['L'+str(i).zfill(2) for i in range(nodes[1])] + ['R'+str(i).zfill(2) for i in range(nodes[2])]
	# print(keys)

	# Height of the local sensor frame above the ground (meters)
	camHeight = 1.62
	# Average distance between two graph nodes (meters)
	avgNodeSpacing = 1
	# Left turn angle (radians)
	leftAngle = 0.5236
	# Right turn angle (radians)
	rightAngle = 0.5236
	# Road-width (meters)
	roadWidth = 4.

	# Generate the OSM graph (implemented as a dictionary)
	graph = {}
	for i in range(nodes[0]-1):
		graph['M'+str(i).zfill(2)] = {'M'+str(i+1).zfill(2): (0, 0, avgNodeSpacing)}
	graph['M'+str(nodes[0]-1).zfill(2)] = {'L'+str(0).zfill(2): (-avgNodeSpacing*np.cos(leftAngle), 0, avgNodeSpacing*np.sin(leftAngle))}
	for i in range(nodes[1]-1):
		graph['L'+str(i).zfill(2)] = {'L'+str(i+1).zfill(2): (0, 0, avgNodeSpacing)}
	graph['M'+str(nodes[0]-1).zfill(2)]['R'+str(0).zfill(2)] = (-avgNodeSpacing*np.cos(rightAngle), 0, avgNodeSpacing*np.sin(rightAngle))
	for i in range(nodes[2]-1):
		graph['R'+str(i).zfill(2)] = {'R'+str(i+1).zfill(2): (0, 0, rightAngle)}
	

	# Simulate road edge points
	road = []
	# Number of points to be sampled along the main branch
	numMain = int(np.floor(nodes[0]*avgNodeSpacing))
	deltaZ = nodes[0]	# Could build flexibility into this
	for i in range(numMain):
		road.append((-0.5*roadWidth, camHeight, deltaZ*i))
	for i in range(numMain):
		road.append((0.5*roadWidth, camHeight, deltaZ*i))
	# Coordinates of left turn corner (and of the right turn corner)
	zleft = deltaZ * numMain
	xleft = -0.5 * roadWidth
	# Number of points to be sampled along the left branch
	numLeft = int(np.floor(nodes[1]*avgNodeSpacing))
	deltaZ = nodes[1]
	for i in range(numLeft):
		mag = deltaZ * i
		road.append((xleft - mag*np.cos(leftAngle), camHeight, zleft + mag*np.sin(leftAngle)))
	# Sample points along the farther edge of the left branch
	# X and Z offsets (corner of the farther point of the left turn intersecting with the projected
	# right edge of the main branch)
	tmpX = xleft + roadWidth * np.sin(leftAngle)
	tmpZ = zleft + roadWidth + roadWidth * np.cos(leftAngle)
	for i in range(numLeft):
		mag = deltaZ * i
		road.append((tmpX - mag*np.cos(leftAngle), camHeight, tmpZ + mag*np.sin(leftAngle)))
	# Number of points to be sampled along the right branch
	numRight = int(np.floor(nodes[2]*avgNodeSpacing))
	deltaZ = nodes[2]
	for i in range(numRight):
		mag = deltaZ * i
		road.append((xleft + roadWidth + mag*np.cos(rightAngle), camHeight, zleft + mag*np.sin(leftAngle)))
	# Sample points along the farther edge of the right branch
	tmpX = xleft + roadWidth - roadWidth*np.sin(rightAngle)
	tmpZ = zleft + roadWidth * np.cos(rightAngle)
	for i in range(numRight):
		mag = deltaZ * i
		road.append((tmpX + mag*np.cos(rightAngle), camHeight, tmpZ + mag*np.sin(rightAngle)))


	plt.scatter([x for (x,y,z) in road], [z for (x,y,z) in road])
	plt.show()

	roadFile = open('road.txt', 'w')
	for (x, y, z) in road:
		roadFile.write(str(x) + ' ' + str(y) + ' ' + str(z) + '\n')
	roadFile.close()
	# np.savetxt('graph.txt')

	osmFile = open('osm.txt', 'w')
	for node, edgeList in graph.items():
		for edge, tf in edgeList.items():
			osmFile.write(node + ' ' + ' ' + edge + ' ' + str(tf[0]) + ' ' + str(tf[1]) + ' ' + str(tf[2]) + '\n')
		# osmFile.write(str(item) + '\n')
	osmFile.close()
