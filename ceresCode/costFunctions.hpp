#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "Eigen/Core"


// Define a struct to hold the distance error. This error specifies that the solution vector
// must be close to the control (initial) vector.
struct DistanceError2D{

	// Constructor
	DistanceError2D(double *road, double *osm) : road_(road), osm_(osm) {}

	// The operator method. Evaluates the cost function and computes the jacobians.
	template <typename T>
	bool operator() (const T* const trans, T* residuals) const {
		residuals[0] = T(1.0)*(osm_[0] + trans[0] - T(road_[0]));
		residuals[1] = T(1.0)*(osm_[1] + trans[1] - T(road_[1]));
		return true;
	}

	// Road point
	double *road_;
	// OSM point
	double *osm_;

};


// Define a struct to hold the regularizer. This error specifies that the solution vector
// must be close to the control (initial) vector.
struct Regularizer2D{

	// Constructor
	Regularizer2D(double *P) : P_(P) {}

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


// Convert angle to 2D rotation matrix (Ceres, unfortunately has no built-in functions for 2D rotations)
template <typename T>
Eigen::Matrix<T, 2, 2> RotationMatrix2D(T yaw_radians) {
	const T cos_yaw = ceres::cos(yaw_radians);
	const T sin_yaw = ceres::sin(yaw_radians);

	Eigen::Matrix<T, 2, 2> rotation;
	rotation << cos_yaw, -sin_yaw, sin_yaw, cos_yaw;
	return rotation;
}


// Define a struct to hold the 2D rigid transform error.
struct SE2Error{

	// Constructor
	SE2Error(double *road, double *osm) : road_(road), osm_(osm);

	// The operator method. Evaluates the cost function and computes the residuals.
	template <typename T>
	bool operator() (const T* const trans, const T* const rot, T* residuals) const {
		T rotated_point = RotationMatrix2D(rot) * T(osm_);
		residuals[0] = T(1.0) * (rotated_point[0] + trans[0] - T(road_[0]));
		residuals[1] = T(1.0) * (rotated_point[1] + trans[1] - T(road_[1]));
	}

	// Road point
	double *road_;
	// OSM point
	double *osm_;

}
