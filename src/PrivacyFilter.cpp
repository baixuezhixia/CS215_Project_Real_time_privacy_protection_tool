#include "PrivacyFilter.hpp"

// ====================== 工具函数：安全 ROI 裁剪 ======================
static cv::Rect safeRect(const cv::Rect& r, const cv::Mat& img) {
  int x = std::max(0, r.x);
  int y = std::max(0, r.y);
  int w = std::min(r.width, img.cols - x);
  int h = std::min(r.height, img.rows - y);
  return cv::Rect(x, y, w, h);
}

// ====================== 高斯模糊 ======================
void PrivacyFilter::blurFace(cv::Mat& frame, const cv::Rect& faceRect,
                             int kernelSize) {
  cv::Rect R = safeRect(faceRect, frame);
  if (R.width <= 0 || R.height <= 0) return;

  cv::GaussianBlur(frame(R), frame(R), cv::Size(kernelSize, kernelSize), 0);
}

// ====================== 像素化 ======================
void PrivacyFilter::pixelateFace(cv::Mat& frame, const cv::Rect& faceRect,
                                 int pixelSize) {
  cv::Rect R = safeRect(faceRect, frame);
  if (R.width <= 0 || R.height <= 0) return;

  int w = std::max(1, R.width / pixelSize);
  int h = std::max(1, R.height / pixelSize);

  cv::Mat temp;
  cv::resize(frame(R), temp, cv::Size(w, h));
  cv::resize(temp, frame(R), R.size(), 0, 0, cv::INTER_NEAREST);
}

// ====================== 猫猫遮挡 ======================
void PrivacyFilter::maskFace(cv::Mat& frame, const cv::Rect& faceRect,
                             const cv::Mat& maskImage) {
  // --- 1. 边界检查 & 裁剪（确保 faceRect 在 frame 内） ---
  cv::Rect roi =
      faceRect & cv::Rect(0, 0, frame.cols, frame.rows);  // intersection
  if (roi.width == 0 || roi.height == 0) return;

  cv::Mat faceROI = frame(roi);

  // --- 2. 如果没有 maskImage 使用默认黑条 ---
  if (maskImage.empty()) {
    int barHeight = roi.height / 5;
    cv::rectangle(frame,
                  cv::Rect(roi.x, roi.y + roi.height / 3, roi.width, barHeight),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    return;
  }
  // --- 3. 缩放 mask 到 ROI 大小（保持 alpha） ---
  cv::Mat resizedMask;
  cv::resize(maskImage, resizedMask, roi.size(), 0, 0, cv::INTER_AREA);

  // 如果没有 alpha 通道，直接覆盖（或可选择用某种阈值生成 alpha）
  if (resizedMask.channels() == 3) {
    resizedMask.copyTo(faceROI);
    return;
  } else if (resizedMask.channels() == 4) {
    for (int y = 0; y < roi.height; ++y) {
      for (int x = 0; x < roi.width; ++x) {
        cv::Vec4b m = resizedMask.at<cv::Vec4b>(y, x);
        float alpha = m[3] / 255.0f;

        cv::Vec3b& bg = faceROI.at<cv::Vec3b>(y, x);

        for (int c = 0; c < 3; ++c) {
          bg[c] = static_cast<uchar>(bg[c] * (1.0f - alpha) + m[c] * alpha);
        }
      }
    }
  }
}
