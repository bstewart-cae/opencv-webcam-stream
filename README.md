# opencv-webcam-stream

### Prerequisites
- CMake >= 3.20
- OpenCV == 4.5.3
### Build and run instructions
```shell
# On Mac OS using CLion's provided CMake to build in Release mode
# e.g., my OpenCV_DIR=/usr/local/Cellar/opencv/4.5.3/lib/cmake/opencv4 under which 'OpenCVConfig.cmake' can be found
cmake -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/usr/local/Cellar/opencv/4.5.3/lib/cmake/opencv4 -DCMAKE_DEPENDS_USE_COMPILER=FALSE -G "CodeBlocks - Unix Makefiles" /Users/spencer/CLionProjects/opencv-webcam-stream/src --build .
make

# Run pixel intensity-based multi-threaded frame presentation application from your webcam (you may need to modify permissions for your IDE shell or Terminal shell application to have access) 
./opencv_webcam_stream
```

In answering the question:
```
 What pixel color format would fit the aforementioned requirements 
 (which are deciding which images to show based measures of pixel intensity based on varying intensity values) 
 best and why?
```

I would say, HSV (Hue, Saturation, Value) since HSV separates the typically more noisy chroma (RGB) information from the intensity information in a manner which allows for convenient access and the ability to perform operations off of pixel intensity information independently, which when I want to decide which images to display based solely off of intensity criteria, makes sense to optimize for. Not to mention, for operating on grayscaled iamges, the V channel of HSV already is the image's grayscale representation.
