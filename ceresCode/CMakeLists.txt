cmake_minimum_required(VERSION 2.8)
project(icp-ceres)

set(Ceres_DIR "/home/jatavalk/libs/ceres-solver/build/lib/cmake/Ceres")
message("-- Check for Ceres solver")
find_package(Ceres REQUIRED)
if(CERES_FOUND)
	message("-- Found Ceres Solver: ${CERES_INCLUDE_DIRS}")
else(CERES_FOUND)
	message(FATAL_ERROR "Cannot find Ceres Solver. Plese set CERES_INCLUDE_DIRS & CERES_LIBRARY")
endif(CERES_FOUND)

include_directories(${CMAKE_INCLUDE_PATH} ${CERES_INCLUDE_DIRS})

# Add the executable
add_executable(topometricRegistration topometricRegistration.cc)
target_link_libraries(topometricRegistration ${CERES_LIBRARIES})
