#include <ceres/local_parameterization.h>
#include "normalize_angle.h"


// Defines a local parameterization for updating the angle to be constrained in [-pi, pi)
class AngleLocalParameterization {
public:

	template <typename T>
	bool operator()(const T* theta_radians, const T* delta_theta_radians, T* theta_radians_plus_delta) const {
		*theta_radians_plus_delta = NormalizeAngle(*theta_radians + *delta_theta_radians);
		return true;
	}

	static ceres::LocalParameterization* Create() {
		return (new ceres::AutoDiffLocalParameterization<AngleLocalParameterization, 1, 1>)
	}
};
