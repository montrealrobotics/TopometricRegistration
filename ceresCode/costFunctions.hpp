#include <ceres/ceres.h>
#include <ceres/rotation.h>


// Define a struct to hold the distance error. This error specifies that the solution vector
// must be close to the control (initial) vector.
struct DistanceError2D{

	// Constructor
	DistanceError2D(double *P) : P_(P) {}

	// The operator method. Evaluates the cost function and computes the jacobians.
	template <typename T>
	bool operator() (const T* const Q, T* residuals) const {
		residuals[0] = T(2)*(Q[0] - T(P_[0]));
		residuals[1] = T(2)*(Q[1] - T(P_[1]));
		return true;
	}

	// Reference cloud (here, road point)
	double *P_;

};

