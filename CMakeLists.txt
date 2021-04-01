cmake_minimum_required(VERSION 3.17)
project(image_stitching)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(image_stitching main.cpp util/base.cpp util/base.h util/filter.h util/filter.cpp detect/edge.h detect/edge.cpp detect/sift.h detect/sift.cpp stitch.h stitch.cpp group.h group.cpp)

target_link_libraries(image_stitching ${OpenCV_LIBS})