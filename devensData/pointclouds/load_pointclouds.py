# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 18:13:07 2018

@author: teddy
"""

import _pickle as pkl
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd

# Change this to the location of the data frame in your system
file_path = 'scans_in_utm_small.pkl'

# df = pkl.load(open(file_path, 'r'))
df = pd.read_pickle(file_path)

# Plot the first pointcloud as a sample
scan = df.iloc[0]['scan_utm']
plt.plot(scan['x'], scan['y'], '.', ms=1)
plt.savefig('/home/jatavalk/code/TopometricRegistration/cache/scan.png')
