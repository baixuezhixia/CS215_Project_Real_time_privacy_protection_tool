#include "PrivacyFilter.hpp"

// ====================== 工具函数：安全 ROI 裁剪 ======================
static cv::Rect safeRect(const cv::Rect& r, const cv::Mat& img)
{
    int x = std::max(0, r.x);
    int y = std::max(0, r.y);
    int w = std::min(r.width,  img.cols - x);
    int h = std::min(r.height, img.rows - y);
    return cv::Rect(x, y, w, h);
}

// ====================== 高斯模糊 ======================
void PrivacyFilter::blurFace(cv::Mat& frame, const cv::Rect& faceRect, int kernelSize)
{
    cv::Rect R = safeRect(faceRect, frame);
    if (R.width <= 0 || R.height <= 0) return;

    cv::GaussianBlur(frame(R), frame(R), cv::Size(kernelSize, kernelSize), 0);
}

// ====================== 像素化 ======================
void PrivacyFilter::pixelateFace(cv::Mat& frame, const cv::Rect& faceRect, int pixelSize)
{
    cv::Rect R = safeRect(faceRect, frame);
    if (R.width <= 0 || R.height <= 0) return;

    int w = std::max(1, R.width  / pixelSize);
    int h = std::max(1, R.height / pixelSize);

    cv::Mat temp;
    cv::resize(frame(R), temp, cv::Size(w, h));
    cv::resize(temp, frame(R), R.size(), 0, 0, cv::INTER_NEAREST);
}

// ====================== 猫猫遮挡 ======================
void PrivacyFilter::maskFace(cv::Mat& frame, const cv::Rect& faceRect, const cv::Mat& maskImage)
{
    // --- 1. 边界检查 & 裁剪（确保 faceRect 在 frame 内） ---
    cv::Rect roi = faceRect & cv::Rect(0, 0, frame.cols, frame.rows); // intersection
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
    cv::resize(maskImage, resizedMask, roi.size(),0,0,cv::INTER_AREA);

    // 如果没有 alpha 通道，直接覆盖（或可选择用某种阈值生成 alpha）
    if (resizedMask.channels() < 4) {
        resizedMask.copyTo(faceROI);
        return;
    }

    // --- 4. 分离通道（B,G,R,A） ---
    std::vector<cv::Mat> ch;
    cv::split(resizedMask, ch); // ch[0]=B, ch[1]=G, ch[2]=R, ch[3]=A

    // --- 5. 将 alpha 归一化到 0..1，并扩展到 3 通道 ---
    cv::Mat alpha;
    ch[3].convertTo(alpha, CV_32FC1, 1.0/255.0); // alpha: CV_32F 单通道，值在 0..1
    cv::Mat alpha3;
    cv::Mat channels[] = {alpha, alpha, alpha};
    cv::merge(std::vector<cv::Mat>{alpha, alpha, alpha}, alpha3); // CV_32FC3

    // --- 6. 准备 src(mask RGB) 与 dst(faceROI) 为 CV_32FC3 ---
    cv::Mat mask_rgb;
    cv::merge(std::vector<cv::Mat>{ch[0], ch[1], ch[2]}, mask_rgb); // CV_8UC3
    cv::Mat mask_f, dst_f;
    mask_rgb.convertTo(mask_f, CV_32FC3); // 0..255
    faceROI.convertTo(dst_f, CV_32FC3);

    // --- 7. 向量化混合： out = mask * alpha + dst * (1-alpha) ---
    cv::Mat out_f = mask_f.mul(alpha3) + dst_f.mul(cv::Scalar(1.0,1.0,1.0) - alpha3);

    // --- 8. 转回 uint8 并写回 ROI ---
    out_f.convertTo(faceROI, CV_8UC3);

}
