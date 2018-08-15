import sys

f = open('osm_nodes.txt')
outFile = open('osm_nodes_noisy.txt', 'w')
trans = [2.0, 1.5]
firstLine = True
for line in f.readlines():
	curLine = line.strip().split()
	if firstLine:
		firstLine = False
		outFile.write(curLine[0] + '\n')
	else:
		outFile.write(str(float(curLine[0])+trans[0]) + ' ' + str(float(curLine[1])+trans[1]) + '\n')
outFile.close()
