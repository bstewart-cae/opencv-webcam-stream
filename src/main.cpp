#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

/**
 * Convert HSV Mat object to BGR
 * @param inputImage
 * @return
 */
Mat hsv2bgr(const Mat &inputImage) {
  Mat fullImageRGB;
  cvtColor(inputImage, fullImageRGB, COLOR_HSV2BGR);
  return fullImageRGB;
}

/**
 * Convert BGR Mat object to HSV
 * @param inputImage
 * @return
 */
Mat bgr2hsv(const Mat &inputImage) {
  Mat fullImageHSV;
  cvtColor(inputImage, fullImageHSV, COLOR_BGR2HSV);
  return fullImageHSV;
}


int main() {
//  std::thread t1;
//  std::thread t2;
  cout << "Doing things with webcam frames..." << endl;
}