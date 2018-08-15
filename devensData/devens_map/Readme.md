# devens_map

The file `devens_map_poly.pkl` contains a pickled polygon representing the road area of the devens test site in [Devens, MA](https://goo.gl/maps/D72TJmmtvQ62). The polygon is an instance of the [Shapely.Geometry.Polygon](http://toblerity.org/shapely/manual.html#polygons) class.

The coordinate frame for the data is `(x,y)=(286273, 4711159)` in UTM coordinates Zone 19T. This frame ensures that the origin (0,0) falls in the test site for convenience.

Run the sample script `python draw_devens_map.py` for a visualization of the polygon, and an example of how to use it to test whether a point is on the road.