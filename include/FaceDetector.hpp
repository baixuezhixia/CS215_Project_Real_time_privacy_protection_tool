#ifndef FACE_DETECTOR_HPP
#define FACE_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect/face.hpp>
#include <string>

class FaceDetector {
public:
    // 构造：传入模型路径与可选参数（阈值、backend/target）
    FaceDetector(const std::string& modelPath,
                 const cv::Size& inputSize = cv::Size(320,320),
                 float confThreshold = 0.9f,
                 float nmsThreshold = 0.3f,
                 int topK = 5000,
                 int backendId = cv::dnn::DNN_BACKEND_OPENCV,
                 int targetId = cv::dnn::DNN_TARGET_CPU);

    ~FaceDetector();

    // 是否成功加载模型
    bool isLoaded() const;

    // 如果摄像头或图像尺寸发生变化，调用此函数设置输入尺寸
    void setInputSize(const cv::Size& size);

    // 对一张图像进行检测，输出 faces（N x 15，跟 YuNet 标准输出一致）
    // 返回检测到的人脸数
    int detect(const cv::Mat& frame, cv::Mat& faces);

    // 可查询/修改阈值
    void setConfThreshold(float t) { confThreshold_ = t; }
    float getConfThreshold() const { return confThreshold_; }

private:
    std::string modelPath_;
    cv::Size inputSize_;
    float confThreshold_;
    float nmsThreshold_;
    int topK_;
    int backendId_;
    int targetId_;

    cv::Ptr<cv::FaceDetectorYN> detector_; // YuNet 专用封装
};

#endif // FACE_DETECTOR_HPP
