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
	const char *filename_constraints;
	const char *filename_output;

	filename_osm = "/home/jatavalk/code/TopometricRegistration/ceresCode/data/osm_nodes_noisy.txt";
	filename_constraints = "/home/jatavalk/code/TopometricRegistration/ceresCode/data/constraints.txt";
	filename_output = "/home/jatavalk/code/TopometricRegistration/ceresCode/data/output.txt";

	TopometricRegistrationProblem2D topometricRegistrationProblem;
	if(!topometricRegistrationProblem.loadFiles(filename_osm, filename_constraints)){
		std::cerr << "ERROR: Unable to open input file(s)" << std::endl;
		return -1;
	}

	// Get references to variables and other metadata

	// Number of OSM nodes
	const int num_osm = topometricRegistrationProblem.getNumOSM();
	// Number of constraints per OSM node
	const int *num_constraints_ = topometricRegistrationProblem.getNumConstraints();
	// Total number of constraints
	const int total_num_constraints = topometricRegistrationProblem.getTotalNumConstraints();
	// OSM nodes
	double *osm = topometricRegistrationProblem.getOSM();
	// Constraints
	double *constraints = topometricRegistrationProblem.getConstraints();

	// Variable to hold the translation vector (translation to be applied to each point
	// on the OSM)
	double trans[2] = {0.1, 0.1};
	double rot = 0.01;

	// Printing out data, for verification
	std::cout << "Num OSM: " << num_osm << std::endl;
	std::cout << "Total num constraints: " << total_num_constraints << std::endl;
	std::cout << "Last OSM point: " << osm[2*num_osm-2] << ", " << osm[2*num_osm-1] << std::endl;
	std::cout << "Last constraint: " << constraints[2*total_num_constraints-2] << ", " << constraints[2*total_num_constraints-1] << std::endl;


	// ----------------------------------
	// Construct the optimization problem
	// ----------------------------------

	// Declare a Ceres optimization problem to hold cost functions
	ceres::Problem problem;

	// Counter, to keep track of constraints added thus far
	int constraint_count = 0;

	// For each OSM-road point pair, add a DistanceError term (2D)
	for(int i = 0; i < num_osm; ++i){
		std::cout << "Num constraints: " << num_constraints_[i] << std::endl;
		for(int j = 0; j < num_constraints_[i]; ++j){
			std::cout << i << ", " << j  << " " << constraint_count << " " << osm[2*i] << " " << osm[2*i+1] << std::endl;
			std::cout << constraints[2*constraint_count] << ", " << constraints[2*constraint_count+1] << std::endl;

			ceres::CostFunction *rigidError2D = new ceres::AutoDiffCostFunction<SE2Error, 2, 1, 2>(
				new SE2Error(constraints + 2*constraint_count, osm + 2*i));
			problem.AddResidualBlock(rigidError2D, new ceres::HuberLoss(0.5), trans, &rot);
			// ceres::CostFunction *distanceError2D = new ceres::AutoDiffCostFunction<DistanceError2D, 2, 2>(
			// 	new DistanceError2D(constraints + 2*constraint_count, osm + 2*i));
			// problem.AddResidualBlock(distanceError2D, new ceres::HuberLoss(0.5), trans);
			constraint_count ++;
		}
	}
	std::cout << "Constraint count: " << constraint_count << std::endl;


	// ----------------------------------
	// Solve the optimization problem
	// ----------------------------------

	// Specify solver options
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	// options.preconditioner_type = ceres::JACOBI;
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
