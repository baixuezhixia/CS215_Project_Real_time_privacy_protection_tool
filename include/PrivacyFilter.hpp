#ifndef PRIVACY_FILTER_HPP
#define PRIVACY_FILTER_HPP

#include <opencv2/opencv.hpp>

class PrivacyFilter {
 public:
  // 高斯模糊
  void blurFace(cv::Mat& frame, const cv::Rect& faceRect, int kernelSize = 35);

  // 像素化
  void pixelateFace(cv::Mat& frame, const cv::Rect& faceRect,
                    int pixelSize = 20);

  // 遮挡（例如黑条或图片）
  void maskFace(cv::Mat& frame, const cv::Rect& faceRect,
                const cv::Mat& maskImage = cv::Mat());
};

#endif  // PRIVACY_FILTER_HPP
