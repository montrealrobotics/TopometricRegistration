# devens_osm

The files `osm_edges.txt` and `osm_nodes.txt` contain the data from the [OpenStreetMap](https://www.openstreetmap.org/#map=14/56.8805/14.8225) of the test site in [Devens, MA](https://goo.gl/maps/D72TJmmtvQ62).

The files each contain two columns of comma separated values where:
* `osm_nodes.txt` has columns (x,y) giving the positions of the nodes, with one node on each line
* `osm_edges.txt` has columns (start_node, end_node) corresponding to the line numbers in `osm_nodes.txt`
 
The coordinate frame for the node location data is `(x,y)=(286273, 4711159)` in UTM coordinates Zone 19T. This frame ensures that the origin (0,0) falls in the test site for convenience.

Run the sample script `python draw_devens_map.py` for a visualization of the polygon, and an example of how to use it to test whether a point is on the road.

Run the sammple script `python draw_graph_from_files.py` for an example of how to load the files into a [networkx](https://networkx.github.io/documentation/stable/reference/classes/graph.html) graph data structure and visualize it.