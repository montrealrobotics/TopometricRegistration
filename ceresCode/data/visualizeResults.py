import matplotlib.pyplot as plt
import sys

# Read in original OSM
file_osm = open('osm_nodes_noisy.txt').readlines()
osm = []
i = 0
for line in file_osm:
	if i == 0:
		i = 1
		pass
	else:
		curLine = line.strip().split()
		osm.append((float(curLine[0]), float(curLine[1])))
# print(len(osm))

file_road = open('road.txt').readlines()
road = []
i = 0
for line in file_road:
	if i == 0:
		i = 1
		pass
	else:
		curLine = line.strip().split()
		road.append((float(curLine[0]), float(curLine[1])))
# print(len(road))

file_output = open('output.txt').readlines()
output = []
i = 0
for line in file_output:
	if i == 0:
		i = 1
		pass
	else:
		curLine = line.strip().split()
		output.append((float(curLine[0]), float(curLine[1])))
# print(len(output))


fig, ax = plt.subplots()
ax.scatter([i for (i,j) in road], [j for (i,j) in road], s = 1, facecolor = 'blue')
ax.scatter([i for (i,j) in osm], [j for (i,j) in osm], s = 1, facecolor = 'green')
plt.savefig('initial.png')

fig, ax = plt.subplots()
ax.scatter([i for (i,j) in road], [j for (i,j) in road], s = 1, facecolor = 'blue')
ax.scatter([i for (i,j) in output], [j for (i,j) in output], s = 1, facecolor = 'red')
plt.savefig('optimized.png')
# plt.show()
