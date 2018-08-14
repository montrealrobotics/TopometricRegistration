import matplotlib.pyplot as plt
import sys

# Read in original OSM
file_osm = open('osm_nodes.txt').readlines()
osm = []
i = 0
for line in file_osm:
	if i == 0:
		i = 1
		pass
	else:
		curLine = line.strip().split()
		osm.append((float(curLine[0]), float(curLine[1])))
print(osm)
