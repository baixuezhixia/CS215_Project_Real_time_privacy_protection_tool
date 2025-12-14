#include "FaceDetector.hpp"
#include <iostream>

FaceDetector::FaceDetector(const std::string& modelPath,
                           const cv::Size& inputSize,
                           float confThreshold,
                           float nmsThreshold,
                           int topK,
                           int backendId,
                           int targetId)
    : modelPath_(modelPath),
      inputSize_(inputSize),
      confThreshold_(confThreshold),
      nmsThreshold_(nmsThreshold),
      topK_(topK),
      backendId_(backendId),
      targetId_(targetId)
{
    try {
        detector_ = cv::FaceDetectorYN::create(
            modelPath_, "", inputSize_,
            confThreshold_, nmsThreshold_, topK_,
            backendId_, targetId_);
    } catch (const cv::Exception& e) {
        std::cerr << "FaceDetector: OpenCV exception while creating detector: " << e.what() << std::endl;
        detector_.release();
    }

    if (detector_.empty()) {
        std::cerr << "FaceDetector: failed to load model: " << modelPath_ << std::endl;
    }
}

FaceDetector::~FaceDetector() {
    // cv::Ptr 会自动释放
}

bool FaceDetector::isLoaded() const {
    return !detector_.empty();
}

void FaceDetector::setInputSize(const cv::Size& size) {
    inputSize_ = size;
    if (!detector_.empty()) {
        // FaceDetectorYN 要求输入尺寸与实际输入匹配
        detector_->setInputSize(inputSize_);
    }
}

int FaceDetector::detect(const cv::Mat& frame, cv::Mat& faces) {
    if (detector_.empty()) {
        faces.release();
        return 0;
    }

    // detector->detect 会根据内部设置返回 N x 15 矩阵
    try {
        detector_->detect(frame, faces);
    } catch (const cv::Exception& e) {
        std::cerr << "FaceDetector::detect OpenCV exception: " << e.what() << std::endl;
        faces.release();
        return 0;
    }

    if (faces.empty()) return 0;
    return faces.rows;
}
