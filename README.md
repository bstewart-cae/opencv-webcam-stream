# opencv-webcam-stream

### Prerequisites
- CMake >= 3.20
- OpenCV == 4.5.3
### Build and run instructions
```shell
# On Mac OS using CLion's provided CMake to build in Release mode
# e.g., my OpenCV_DIR=/usr/local/Cellar/opencv/4.5.3/lib/cmake/opencv4 under which 'OpenCVConfig.cmake' can be found
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/usr/local/Cellar/opencv/4.5.3/lib/cmake/opencv4 -DCMAKE_DEPENDS_USE_COMPILER=FALSE -G "CodeBlocks - Unix Makefiles" /Users/spencer/CLionProjects/opencv-webcam-stream ..
make

# Run pixel intensity-based multi-threaded frame presentation application from your webcam (you may need to modify permissions for your IDE shell or Terminal shell application to have access) 
./opencv_webcam_stream
```
