#include "opencv2/opencv.hpp"
#include <stdexcept>
#include <iostream>
#include <thread>
#include <chrono>
#include <queue>
#include <mutex>
#include <atomic>

using namespace cv;
using namespace std;
using namespace std::chrono_literals;

static atomic<bool> done = ATOMIC_FLAG_INIT;
static const string T1_WINDOW_NAME = "t1 window";
static const string T2_WINDOW_NAME = "t2 window";

/* START Overhead dependencies and utilities */
/**
 * A thread-safe queue
 * @tparam T
 */
template<class T>
class ThreadSafeQueue {
private:
    queue<T> queue_;
    mutable mutex mutex_;
    condition_variable cond_;
public:
    ThreadSafeQueue()
            : queue_(), mutex_(), cond_() {}

    ~ThreadSafeQueue() = default;

    /**
     * Get the current size of the queue
     */
    unsigned long size() {
      unique_lock<mutex> lock(mutex_);
      unsigned long size = queue_.size();
      lock.unlock();
      cond_.notify_one();
      return size;
    }

    bool is_accessible() {
      bool locked = mutex_.try_lock();
      mutex_.unlock();
      cond_.notify_one();
      return locked;
    }

    unique_lock<mutex> reserve() {
      unique_lock<mutex> lock(mutex_);
      cond_.notify_one();
      return lock;
    }

    bool relinquish(unique_lock<mutex> &lock_) {
      lock_.unlock();
      cond_.notify_one();
      return is_accessible();
    }

    /**
     * Add an element to the queue
     */
    void enqueue(T elem) {
      lock_guard<mutex> lock(mutex_);
      queue_.push(elem);
      cond_.notify_one();
    }

    /**
     * Grab the element at the front of the queue, otherwise, wait until
     * an element becomes available
     * @return
     */
    T dequeue() {
      unique_lock<mutex> lock(mutex_);

      // release lock to grab it again once waiting period is up
      cond_.wait(lock, [&] {
          return !queue_.empty();
      });

      T val = queue_.front();
      queue_.pop();
      return val;
    }

    /**
     * Without extracting or returning item at front of the queue, remove it from the queue
     * if the queue isn't empty (single-shot)
     */
    void pop() {
      lock_guard<mutex> lock(mutex_);
      if (!queue_.empty())
        queue_.pop();
    }

    /**
     * Grab front element of queue without removing it from the queue
     * @return
     */
    T front() {
      unique_lock<mutex> lock(mutex_);

      // release lock to grab it again once waiting period is up
      cond_.wait(lock, [&] {
          return !queue_.empty();
      });

      T val = queue_.front();
      return val;
    }
};

/**
 * Convert HSV Mat object to BGR
 * @param inputImage
 * @param fullImageBGR value passed by-reference
 * @return
 */
void hsv2bgr(const Mat &inputImage, Mat &fullImageBGR) {
  cvtColor(inputImage, fullImageBGR, COLOR_HSV2BGR);
}

/**
 * Convert BGR Mat object to HSV
 * @param inputImage
 * @param fullImageHSV value passed by-reference
 * @return
 */
void bgr2hsv(const Mat &inputImage, Mat &fullImageHSV) {
  cvtColor(inputImage, fullImageHSV, COLOR_BGR2HSV);
}

/**
 * Pull out grayscale information from HSV image
 * @param inputImage
 * @param fullImageGray
 */
void hsv2gray(const Mat &inputImage, Mat &fullImageGray) {
  Mat hsv_channels[3];
  cv::split(inputImage, hsv_channels);
  fullImageGray = hsv_channels[2];
}

/**
 * Rotate image
 * @param frame Mat object which represents pre-rotated image
 * @param output Mat object to update as rotated image
 * @param angle positive (counterclockwise), negative (clockwise) angle to rotate image based upon
 */
void rotate_image(const Mat &frame, Mat &output, double angle) {
  // obtain the rotation matrix in order to rotate image around the pixel center
  Point2f center(float((frame.cols - 1) / 2.0), float((frame.rows - 1) / 2.0));
  Mat rotMatrix = getRotationMatrix2D(center, angle, 1.0);
  Rect2f bbox = cv::RotatedRect(cv::Point2f(), frame.size(), float(angle)).boundingRect2f();

  // adjust the transformation matrix
  rotMatrix.at<double>(0, 2) += bbox.width / 2.0 - frame.cols / 2.0;
  rotMatrix.at<double>(1, 2) += bbox.height / 2.0 - frame.rows / 2.0;

  warpAffine(frame, output, rotMatrix, bbox.size());
}

/* END Overhead dependencies and utilities */

void process_frame_mode_0(ThreadSafeQueue<Mat> &frame_queue, ThreadSafeQueue<Mat> &display_queue, atomic<bool> &done) {
  Mat new_frame, grayscale_frame;
  unsigned int num_frames_pushed = 0;
  double image_angle = 0;

  while (!done.load(memory_order_relaxed)) {
    // check if the queue is as full as we want it (10 frames)
    if (frame_queue.size() < 10) {
      // remove and grab frame reference off of the queue
      new_frame = frame_queue.dequeue();
    } else {
      // skip oldest frame and continue on since we are trying to operate above 10 frames in the queue
      frame_queue.pop();
      continue;
    }

    num_frames_pushed++;

    if (num_frames_pushed % 5 == 0) {
      // rotate image 90 degrees clockwise
      image_angle -= 90;
      // if rotation update is back to 360, update image angle to 0 to indicate no rotation needed
      if (image_angle == -360)
        image_angle = 0;
    }

    // pull out grasycale channel from HSV
    hsv2gray(new_frame, grayscale_frame);

    // when needing to displayed rotated image, rotate the image to be displayed
    if (image_angle < 0) {
      rotate_image(grayscale_frame, grayscale_frame, image_angle);
    }

    // queue display of frame
    display_queue.enqueue(grayscale_frame);
  }
}

void process_frame_mode_1(ThreadSafeQueue<Mat> &frame_queue, ThreadSafeQueue<Mat> &display_queue, atomic<bool> &done) {
  Mat hsv_frame, grayscale_frame, bgr_frame, mirrored, left_half, hist;

  while (!done.load(memory_order_relaxed)) {
    // check if the queue has 1 frame to process
    auto top = chrono::steady_clock::now();

    unsigned long queue_size = frame_queue.size();
    if (queue_size != 1) {
      if (queue_size > 1) {
        cout
                << "Something likely went wrong... There shouldn't be more than one frame queued at a time. Current size is: "
                << queue_size << endl;
      }
      continue;
    }

    // hold lock on queue while processing around queue specific operations even
    unique_lock<mutex> queue_lock = frame_queue.reserve();

    // peek and grab current (oldest) frame in queue (the only at this point), but don't remove it yet
    hsv_frame = frame_queue.front();

    int vbins = 16;
    int histSize[] = {vbins};

    float vranges[] = {0, 256};
    const float *ranges[] = {vranges};

    int channels[] = {0};

    // extract pure grayscale from HSV
    hsv2gray(hsv_frame, grayscale_frame);

    left_half = grayscale_frame(Rect(0, 0, grayscale_frame.cols / 2, grayscale_frame.rows));

    // compute grayscale histogram
    calcHist(&left_half, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);

    // normalize histogram values
    normalize(hist, hist, 0, 512, NORM_MINMAX, -1, Mat());

    Scalar tempVal = mean(hist);

    // take the mean of all histogram values, divide that by the number of bins, and then by the histogram step size
    auto mean_pixel_intensity = float(tempVal[0]);

    // get original image
    hsv2bgr(hsv_frame, bgr_frame);

    // now that we are done processing, remove the item from the queue
    // so other thread users of thread-safe queue can appropriately check the size before inserting a new frame
    frame_queue.pop();

    // max pixel intensity is 255, check if mean pixel intensity is > half of that
    if (mean_pixel_intensity > 127.5) {
      mirrored = Mat(bgr_frame.rows, bgr_frame.cols, CV_8UC3);
      flip(bgr_frame, mirrored, 1);

      // queue display of mirrored frame
      display_queue.enqueue(mirrored);
    } else {
      // queue display of original frame
      display_queue.enqueue(bgr_frame);
    }

    // relinquish a lock hold since all processing of previous frame is done
    frame_queue.relinquish(queue_lock);

    auto ms = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - top).count();

    // sleep until second is up since last processing began
    this_thread::sleep_for(chrono::milliseconds(1000 - ms));
  }
}

/**
 * Controller thread function which manages frame distribution for processing
 * @param cam
 * @param done
 * @param frame_queue_0
 * @param frame_queue_1
 */
void controller(VideoCapture &cam, atomic<bool> &done, ThreadSafeQueue<Mat> &frame_queue_0,
                ThreadSafeQueue<Mat> &frame_queue_1) {
  Mat frame, hsv_frame;

  unsigned int frame_idx = 0;

  while (!done.load(memory_order_relaxed)) {
    // get new frame
    cam >> frame;

    // if empty frame was pulled, skip it (rare, but possible)
    if (frame.empty())
      continue;

    // convert BGR frame to HSV
    bgr2hsv(frame, hsv_frame);

    // For t1:
    // since frame index is zero-based, push every 2nd frame to t1 by checking for an odd index
    if (frame_idx % 2 != 0)
      frame_queue_0.enqueue(hsv_frame);

    // For t2:
    // push frame to t2 via 'frame_queue_1' effectively only as close to every 1 second as possible
    if (frame_queue_1.is_accessible() && frame_queue_1.size() == 0)
      frame_queue_1.enqueue(hsv_frame);

    // increment frame index
    frame_idx++;
  }
}

int main() {
  VideoCapture cam(0);

  if (!cam.isOpened()) {
    throw runtime_error("Could not properly begin video capture");
  }

  // declare Mat frame queues for each of the 2 window-dedicated threads
  ThreadSafeQueue<Mat> frame_queue_0, frame_queue_1;

  // declare Mat display queues for each of the 2 window-dedicated threads
  ThreadSafeQueue<Mat> display_queue_0, display_queue_1;

  // define atomic flag to indicate key press to stop capture, frame feeding and displaying
  done.store(false);

  // initialize named windows for t1 and t2 outputs
  namedWindow(T1_WINDOW_NAME, WINDOW_AUTOSIZE);
  namedWindow(T2_WINDOW_NAME, WINDOW_AUTOSIZE);

  std::thread controller_(controller, ref(cam), ref(done), ref(frame_queue_0), ref(frame_queue_1));

  std::thread t1(process_frame_mode_0, ref(frame_queue_0), ref(display_queue_0), ref(done));
  std::thread t2(process_frame_mode_1, ref(frame_queue_1), ref(display_queue_1), ref(done));

  Mat frame_to_display_0, frame_to_display_1;

  while (!done.load(memory_order_relaxed)) {
    // handle window display updates
    frame_to_display_0 = display_queue_0.dequeue();
    frame_to_display_1 = display_queue_1.dequeue();

    // update display windows
    imshow(T1_WINDOW_NAME, frame_to_display_0);
    imshow(T2_WINDOW_NAME, frame_to_display_1);

    // handle keyboard events
    int c = waitKey(1);
    if (c == 27)
      done.store(true, memory_order_relaxed);
  }

  // join all threads which should have terminated because atomic flag updates
  cout << "Joining threads..." << endl;
  controller_.join();

  // once controller thread exits, cam object can be safely interacted with
  cam.release();

  t1.join();
  t2.join();

  // OpenCV capture (invalidating capture object reference that '_controller' was using) and display window cleanup
  cout << "Stopping capture and closing out all windows..." << endl;
  destroyWindow(T1_WINDOW_NAME);
  destroyWindow(T2_WINDOW_NAME);
}