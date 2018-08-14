#include <ceres/ceres.h>
#include <ceres/rotation.h>


// Class that holds data and functions required for the topometric registration problem
class TopometricRegistrationProblem2D{

public:

	// Get a pointer to the OSM
	double *getOSM() { return osm_; }

	// Get a pointer to the road
	double *getRoad() { return road_; }

	// Get number of OSM points
	int getNumOSM() { return num_osm_; }

	// Get number of road points
	int getNumRoad() { return num_road_; }

	// Read data from input files
	bool loadFiles(const char *filename_osm, const char *filename_road){
		
		FILE *fptr_osm = fopen(filename_osm, "r");
		if(fptr_osm == NULL){
			std::cout << "Unable to open " << std::string(filename_osm) << std::endl;
			return false;
		}
		fscanfOrDie(fptr_osm, "%d", &num_osm_);
		osm_ = new double[2*num_osm_];
		for(int i = 0; i < num_osm_; ++i)
			for(int j = 0; j < 2; ++j)
				fscanfOrDie(fptr_osm, "%lf", osm_ + i*2 + j);

		FILE *fptr_road = fopen(filename_road, "r");
		if(fptr_road == NULL){
			std::cout << "Unable to open " << std::string(filename_road) << std::endl;
			return false;
		}
		fscanfOrDie(fptr_road, "%d", &num_road_);
		road_ = new double[2*num_road_];
		for(int i = 0; i < num_road_; ++i)
			for(int j = 0; j < 2; ++j)
				fscanfOrDie(fptr_road, "%lf", road_ + i*2 + j);

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
	double *road_;
	int num_osm_, num_road_;

};
