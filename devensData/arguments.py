import argparse

def get_args():
	
	parser = argparse.ArgumentParser(description='call your parameters')

	parser.add_argument('--lamda', type=float, default=0.01, help="weight for cost2 between OSM nodes")
	parser.add_argument('--opti', type=str, default='powell', help="powell / nelder-mead")
	parser.add_argument('--osm-thresh', type=float, default=100.0, help="radius from which OSM nodes should be sampled")

	args = parser.parse_args()

	return args