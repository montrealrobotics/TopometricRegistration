#include <ceres/ceres.h>
#include <ceres/rotation.h>


// Class that holds data and functions required for the topometric registration problem
class TopometricRegistrationProblem2D{

public:

	// Get a pointer to the OSM
	double *getOSM() { return osm_; }

	// Get a pointer to the constraints
	double *getConstraints() { return constraints_; }

	// Get number of OSM points
	int getNumOSM() { return num_osm_; }

	// Get number of road points
	int *getNumConstraints() { return num_constraints_; }

	// Get the total number of constraints
	int getTotalNumConstraints() { return total_num_constraints_; }

	// Read data from input files
	bool loadFiles(const char *filename_osm, const char *filename_constraints){
		
		// Load OSM
		FILE *fptr_osm = fopen(filename_osm, "r");
		if(fptr_osm == NULL){
			std::cout << "Unable to open " << std::string(filename_osm) << std::endl;
			return false;
		}
		fscanfOrDie(fptr_osm, "%d", &num_osm_);
		osm_ = new double[2*num_osm_];
		for(int i = 0; i < num_osm_; ++i){
			for(int j = 0; j < 2; ++j){
				fscanfOrDie(fptr_osm, "%lf", osm_ + i*2 + j);
			}
		}

		// Load constraints
		FILE *fptr_constraints = fopen(filename_constraints, "r");
		if(fptr_constraints == NULL){
			std::cout << "Unable to open " << std::string(filename_constraints) << std::endl;
			return false;
		}
		int num_constraint_classes_;
		fscanfOrDie(fptr_constraints, "%d", &num_constraint_classes_);
		std::cout << "num_constraint_classes_: " << num_constraint_classes_ << std::endl;
		num_constraints_ = new int[num_constraint_classes_];
		
		total_num_constraints_ = 0;
		for(int i = 0; i < num_constraint_classes_; ++i){
			fscanfOrDie(fptr_constraints, "%d", num_constraints_ + i);
			total_num_constraints_ += num_constraints_[i];
			std::cout << num_constraints_[i] << " ";
		}
		std::cout << std::endl;
		std::cout << total_num_constraints_ << std::endl;
		constraints_ = new double[2*total_num_constraints_];

		int elapsed_constraints_ = 0;
		for(int i = 0; i < total_num_constraints_; ++i){
			for(int j = 0; j < 2; ++j){
				fscanfOrDie(fptr_constraints, "%lf", constraints_ + i*2 + j);
			}
		}

		return true;
	}


private:

	// Helper function to read in one value to a text file
	template <typename T>
	void fscanfOrDie(FILE *fptr, const char *format, T *value){
		int numScanned = fscanf(fptr, format, value);
		if(numScanned != 1){
			LOG(FATAL) << "Invalid data file";
		}
	}

	// Private variables and functions here
	double *osm_;
	double *constraints_;
	int num_osm_;
	int *num_constraints_;
	int total_num_constraints_;

};
