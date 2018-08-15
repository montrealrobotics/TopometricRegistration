# pointclouds

The file `scans_in_utm_small.pkl` contains a pickled table of LiDAR scans taken in the Blue Prius test vehicle in [Devens, MA](https://goo.gl/maps/D72TJmmtvQ62). 

Run the sample script `python load_pointclouds.py` for an example of how to load the data and visualize one scan.

This 'small' file only contains two scans, but is identical in structure to the file `scans_in_utm.pkl` which contains over 3000 scans. That file is ~13GB so it will not be stored in this repo. Please email for access.

The table is an instance of the [Pandas.DataFrame](https://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe) class with columns:
*`lat:` Latitude of the vehicle when the scan was taken
*`lon:` Longitude of the vehicle when the scan was taken
*`scan:` A [numpy.recarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html) containing the LiDar data in the sensor frame. (See Scan Data section below)
*`theta_imu:` The angle (in radians) of the vehicle IMU when the scan was taken
*`x:` The `x-coordinate` of the vehicle in the local devens frame (See Devens Frame section below)
*`y:` The `y-coordinate` of the vehicle in the local devens frame (See Devens Frame section below)
*`theta:` The angle of the vehicle (in radians) with a 0.0 -> x-axis convention
*`scan_utm:` A [numpy.recarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html) containing the LiDar data in devens frame. (See Devens Frame section below)

### Devens Frame
The devens coordinate frame is `(x,y)=(286273, 4711159)` in UTM coordinates Zone 19T. This frame ensures that the origin (0,0) falls in the test site for convenience.

### Scan Data
Each scan is stored as a [numpy.recarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html) with the following columns: `(x,y,z,intensity,ring)`. The `x,y,z` columns give the locations of each 3D point. The `intensity` column gives an (uncalibrated) intensity measurement indicating the surface reflectivity, and the `ring` columns gives an integer in the range [0-63] indicating with which of the 64 laser sensors the point was measured.