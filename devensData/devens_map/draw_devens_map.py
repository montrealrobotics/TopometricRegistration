# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 23:43:13 2018

@author: teddy
"""

# import cPickle as pkl
import _pickle as pkl
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import shapely.geometry as geo

#%% Load the polygon
devens = pkl.load(open('devens_map_poly.pkl', 'rb'), encoding = 'bytes')


#%% Show
plot_coords = lambda obj: plt.plot(obj.xy[0],obj.xy[1], 'k')
plot_coords(devens.exterior)
[plot_coords(x) for x in devens.interiors]

#%% Check some points
print("Is the origin on the road? {}".format(devens.contains(geo.Point([0,0]))))
plt.plot(0,0,'gx')

print("Is (10,10) on the road? {}".format(devens.contains(geo.Point([10,10]))))
plt.plot(10,10,'rx')
plt.savefig('/home/jatavalk/code/TopometricRegistration/cache/devens_map.png')
