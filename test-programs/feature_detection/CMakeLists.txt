cmake_minimum_required(VERSION 2.8)
project( featurepoints )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( featurepoints featurepoints.cpp )
target_link_libraries( featurepoints ${OpenCV_LIBS} )