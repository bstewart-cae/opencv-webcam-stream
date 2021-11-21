cmake_minimum_required(VERSION 3.20)
project(opencv_webcam_stream)

set(OpenCV_DIR /usr/local/Cellar/opencv/4.5.3/lib/cmake/opencv4)

FIND_PACKAGE( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )
LINK_DIRECTORIES( ${OpenCV_LIBS_DIR} )

set(CMAKE_CXX_STANDARD 14)

add_executable(opencv_webcam_stream main.cpp)
TARGET_LINK_LIBRARIES(opencv_webcam_stream ${OpenCV_LIBS})