#include "opencv2/opencv.hpp"
#include <thread>

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

void process_frame_mode_0(const queue<Mat> &frame_queue) {
  while (true) {
    size_t num_frames_in_queue = frame_queue.size();

  }
}

void process_frame_mode_1(const queue<Mat> &frame_queue) {
  while (true) {
    size_t num_frames_in_queue = frame_queue.size();

  }
}


int main() {
  cv::VideoCapture cam(0);

  if (!cam.isOpened()) {
    throw std::runtime_error("Error");
  }

  cv::namedWindow("Window");
  cv::Mat output(350,350,CV_8UC1);
  cv::Mat rgb_output(350,350,CV_8UC3);

//  while(true){
  cv::Mat frame;
  cam>>frame;
  cv::resize(frame, frame, cv::Size(350, 350));

  cv::imshow("bgr_frame", frame);
  std::queue<Mat> frame_queue;
  std::thread t1(process_frame_mode_0, frame_queue);
  std::thread t2(process_frame_mode_1, frame_queue);
  cout << "Doing things with webcam frames..." << endl;
}