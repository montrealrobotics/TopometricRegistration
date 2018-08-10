import numpy as np
import matplotlib.pyplot as plt
import operator
import csv


def sample(x, mu, stddev=0.01):
    return x + np.random.normal(mu, stddev)

def generate_turn(info, laser_res, osm_res, road_width, last_osm, idx, left=True):
    cur_x = 0
    cur_osm_x = 0
    cur_osm = None

    points = []
    osm = []
    osm_connectivity = []

    # Add road points
    while abs(cur_x) < info['dist']:
        if left:
            l_sample = sample(cur_x, mu=0.01, stddev=0.05) * info['slope'] + info['start']
            r_sample = sample(cur_x, mu=0.01, stddev=0.05) * info['slope'] + info['start'] + road_width
            
            points.append((cur_x, l_sample))
            points.append((cur_x, r_sample))

            cur_x -= laser_res * np.abs(np.random.normal(laser_res))

        else:
            l_sample = sample(cur_x, mu=0.01, stddev=0.05) * info['slope'] + info['start'] + road_width
            r_sample = sample(cur_x, mu=0.01, stddev=0.05) * info['slope'] + info['start']

            points.append((cur_x + road_width, l_sample))
            points.append((cur_x + road_width, r_sample)) 

            cur_x += laser_res * np.abs(np.random.normal(laser_res))

    # Add OSM information
    while abs(cur_osm_x) < info['dist']:
        if left:
            osm_sample = sample(cur_osm_x, mu=0.01, stddev=0.01) * info['slope'] + info['start'] + road_width / 2
            cur_osm = 'L{}'.format(str(idx).zfill(2))

            osm.append((cur_osm, cur_osm_x, osm_sample))
            cur_osm_x -= osm_res
        else:
            osm_sample = sample(cur_osm_x, mu=0.01, stddev=0.01) * info['slope'] + info['start'] + road_width / 2
            cur_osm = 'R{}'.format(str(idx).zfill(2))
            
            osm.append((cur_osm, cur_osm_x + road_width, osm_sample))
            cur_osm_x += osm_res

        idx += 1
        osm_connectivity.append((last_osm, cur_osm))
        last_osm = cur_osm

    return points, osm, osm_connectivity, idx


def generate_graph(road_info):
    # Constants
    laser_res = 0.01
    osm_res = 0.5 
    road_width = 1.0

    # Current z coordinates
    cur_l_z = 0
    cur_r_z = 0
    cur_osm_z = 0

    # Indices to keep track of 
    M = 1
    R = 0
    L = 0

    # Items of interest
    laser_points = []
    osm_points = []
    osm_connectivity = []

    # Add M0
    osm_points.append(('M00', road_width / 2, 0))
    last_osm = 'M00'

    for info in road_info:

        # Add OSM Nodes 
        while cur_osm_z < info['start']:
            sample_z = sample(cur_osm_z, mu=osm_res)
            cur_osm = 'M{}'.format(str(M).zfill(2))

            osm_points.append((cur_osm, road_width / 2, sample_z))

            osm_connectivity.append((last_osm, cur_osm))
            cur_osm_z = sample_z

            M += 1
            last_osm = cur_osm

        # If the first item is a left turn
        if info['turn'] == 'left':
            # Add straight road points until we hit the start of the turn
            while cur_l_z < info['start']:
                sample_z = sample(cur_l_z, mu=0.01, stddev=0.005)

                if sample_z < info['start']:
                    laser_points.append((sample(0, laser_res), sample_z))
                
                cur_l_z = sample_z
            
            # Generate the points and OSM information for the turn 
            points, osm, osm_connects, L = generate_turn(info, laser_res, osm_res, road_width, last_osm, L)
            cur_l_z = info['start'] + road_width

        else:
            # Add straight road points until we hit the start of the turn
            while cur_r_z < info['start']:
                sample_z = sample(cur_r_z, mu=0.01, stddev=0.005)

                if sample_z < info['start']:
                    laser_points.append((sample(road_width, laser_res), sample_z))

                cur_r_z = sample_z
            # Generate the points and OSM information for the turn 
            points, osm, osm_connects, R = generate_turn(info, laser_res, osm_res, road_width, last_osm, R, left=False)
            cur_r_z = info['start'] + road_width

        laser_points += points
        osm_points += osm
        osm_connectivity += osm_connects

    # For competeness sake, make sure both parts of the straight road end at the same place
    if cur_l_z < cur_r_z:
        while cur_l_z < cur_r_z:
            sample_z = sample(cur_l_z, mu=0.01, stddev=0.005)
            if sample_z < cur_r_z:
                laser_points.append((sample(0, laser_res), sample_z))

            cur_l_z = sample_z
    else:
        while cur_r_z < cur_l_z:
            sample_z = sample(cur_r_z, mu=0.01, stddev=0.005)

            if sample_z < cur_l_z:
                laser_points.append((sample(road_width, laser_res), sample_z))

            cur_r_z = sample_z

    osm_points_plot = [o[1:] for o in osm_points]

    # Gather required information
    with open('osm_connectivity.csv', 'w') as f:
        file_writer = csv.writer(f)
        for i in range(len(osm_connectivity)):
            file_writer.writerow(osm_connectivity[i])

    with open('road.csv', 'w') as f:
        file_writer = csv.writer(f)
        for i in range(len(laser_points)):
            file_writer.writerow(laser_points[i])

    with open('osm_nodes.csv', 'w') as f:
        file_writer = csv.writer(f)
        for i in range(len(osm_points)):
            file_writer.writerow(osm_points[i])

    # Plot the points we've generated
    plt.scatter(*zip(*laser_points))
    plt.scatter(*zip(*osm_points_plot))
    plt.show()


if __name__ == '__main__':
    road_info = [
        {
            'start': 2.0, 
            'slope': -1, 
            'dist': 2,
            'turn': 'left'
        },
        {
            'start': 2.5, 
            'slope': 0.75, 
            'dist': 2,
            'turn': 'right'
        },
        {
            'start': 5.0, 
            'slope': 0.5, 
            'dist': 1,
            'turn': 'left'
        },

        {
            'start': 7.0, 
            'slope': 0.5, 
            'dist': 1,
            'turn': 'right'
        }
    ]

    generate_graph(road_info)


