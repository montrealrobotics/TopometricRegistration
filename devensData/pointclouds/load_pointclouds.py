# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 18:13:07 2018

@author: teddy
"""

import cPickle as pkl
import matplotlib.pyplot as plt

# Change this to the location of the data frame in your system
file_path = 'scans_in_utm_small.pkl'

df = pkl.load(open(file_path, 'r'))

# Plot the first pointcloud as a sample
scan = df.iloc[0]['scan_utm']
plt.plot(scan['x'], scan['y'], '.', ms=1)
plt.show()
