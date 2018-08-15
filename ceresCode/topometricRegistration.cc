/*
 * Pose adjustmer using keypoint likelihoods from a single image
 * Author: Krishna Murthy
*/

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>

#include <ceres/loss_function.h>
#include <ceres/iteration_callback.h>
#include <ceres/rotation.h>

// Contains definitions of various problem structs
#include "problemStructs.hpp"
// Contains various cost function struct specifications
#include "costFunctions.hpp"


int main(int argc, char** argv){

	google::InitGoogleLogging(argv[0]);

	const char *filename_osm;
	const char *filename_road;
	const char *filename_output;

	filename_osm = "../data/osm_nodes.txt";
	filename_road = "../data/road.txt";
	filename_output = "../data/output.txt";

	TopometricRegistrationProblem2D topometricRegistrationProblem;
	if(!topometricRegistrationProblem.loadFiles(filename_osm, filename_road)){
		std::cerr << "ERROR: Unable to open input file(s)" << std::endl;
		return -1;
	}

	// Get references to variables and other metadata
	const int num_osm = topometricRegistrationProblem.getNumOSM();
	const int num_road = topometricRegistrationProblem.getNumRoad();
	double *osm = topometricRegistrationProblem.getOSM();
	double *road = topometricRegistrationProblem.getRoad();

	// Variable to hold the translation vector (translation to be applied to each point
	// on the OSM)
	double trans[2] = {0.000001, 0.000001};

	// Printing out data, for verification
	std::cout << "Num OSM: " << num_osm << std::endl;
	std::cout << "Num road: " << num_road << std::endl;
	std::cout << "Last OSM point: " << osm[2*num_osm-2] << ", " << osm[2*num_osm-1] << std::endl;
	std::cout << "Last road point: " << road[2*num_road-2] << ", " << road[2*num_road-1] << std::endl;


	// ----------------------------------
	// Construct the optimization problem
	// ----------------------------------

	// Declare a Ceres optimization problem to hold cost functions
	ceres::Problem problem;

	// For each OSM-road point pair, add a DistanceError term (2D)
	for(int i = 0; i < num_road; ++i){
		// ceres::CostFunction *distanceError2D = new ceres::AutoDiffCostFunction<DistanceError2D, 2, 2>(
		// 	new DistanceError2D(road + 2*i));
		for(int j = 0; j < num_osm; ++j){
			// ceres::CostFunction *distanceError2D = new ceres::AutoDiffCostFunction<DistanceError2D, 2, 2>(
			// new DistanceError2D(road + 2*i));
			// problem.AddResidualBlock(distanceError2D, new ceres::HuberLoss(1), osm + 2*j);
			ceres::CostFunction *distanceError2D = new ceres::AutoDiffCostFunction<DistanceError2D, 2, 2>(
				new DistanceError2D(road + 2*i, osm + 2*j));
			problem.AddResidualBlock(distanceError2D, new ceres::HuberLoss(1.0), trans);
		}
	}


	// ----------------------------------
	// Solve the optimization problem
	// ----------------------------------

	// Specify solver options
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.preconditioner_type = ceres::JACOBI;
	options.minimizer_progress_to_stdout = true;

	// Solve the problem and print the results
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;


	// Print estimate
	std::cout << "Translation: " << trans[0] << " " << trans[1] << std::endl;

	// Open an output stream, to write the result to file
	std::ofstream outFile;
	outFile.open(filename_output);
	outFile << num_osm << std::endl;
	for(int i = 0; i < num_osm; ++i){
		outFile << osm[2*i] + trans[0] << " " << osm[2*i + 1] + trans[1] << std::endl;
	}


 	return 0;

}
