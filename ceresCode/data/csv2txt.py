import sys

f = open('road.txt')
outFile = open('road_new.txt', 'w')
firstLine = True
for line in f.readlines():
	curLine = line.strip().split(',')
	if firstLine:
		firstLine = False
		outFile.write(curLine[0] + '\n')
	else:
		outFile.write(curLine[0] + ' ' + curLine[1] + '\n')
outFile.close()
