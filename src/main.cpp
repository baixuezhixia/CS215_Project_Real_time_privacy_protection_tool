#include <opencv2/opencv.hpp>
#include <iostream>
#include "FaceDetector.hpp"
#include "PrivacyFilter.hpp"
//-----------------------[ATTENTION]----------------------------------
//编译运行之前需要先绑定摄像头（bind会自动取消）
//在powershell中运行以下命令
//usbipd attach --busid 2-7 --wsl

//运行前记得删除之前编译的版本
//rm yunet
//编译
/*
g++ src/*.cpp -Iinclude `pkg-config --cflags --libs opencv4` -std=c++17 -o yunet
*/
//运行
//./yunet
//----------------------------------------------------------------------

int main(int argc, char** argv)
{

    int mode =0;//默认“None”

    // ---- 1. 参数设置（对齐标准实现的默认参数）----
    std::string modelPath = "assets/face_detection_yunet_2023mar.onnx"; // 模型路径
    //加载猫猫
    cv::Mat maskImg = cv::imread("assets/cat_mask.png",cv::IMREAD_UNCHANGED);

    // 创建 FaceDetector
    FaceDetector detector(
        modelPath,
        cv::Size(320,320),   // 输入尺寸（初始化时会设置）
        0.6f,                // confThreshold
        0.3f,                // nmsThreshold
        5000                 // topK
    );


    if (!detector.isLoaded()) {
        std::cerr << "❌ 模型加载失败： " << modelPath << std::endl;
        return -1;
    }
    std::cout << "✅ YuNet 模型加载成功\n";

    //创建 PrivacyFilter
    PrivacyFilter filter;
    // ---- 2. 打开摄像头 ----
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    
    if (!cap.isOpened()) {
        std::cerr << "❌ 摄像头打开失败\n";
        return -1;
    }


    // 获取实际分辨率，传给FaceDetector
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    detector.setInputSize(cv::Size(frameWidth, frameHeight));

    cv::Mat frame, faces;
    cv::TickMeter tick; // 用于计算 FPS

    //---- 3.主循环----
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        tick.start();
        int faceCount = detector.detect(frame,faces);
        tick.stop();
        //---- 4.显示FPS----
        float fps = tick.getFPS();
        cv ::putText(frame,cv::format("FPS: %.2f",fps),
                     cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX,
                     1.0, cv::Scalar(0,255,0),2);
        //---- 显示当前隐私模式----
        std::string modeText;
        if (mode == 0) modeText = "Mode 0: None";
        if (mode == 1) modeText = "Mode 1: Blur";
        if (mode == 2) modeText = "Mode 2: Pixelate";
        if (mode == 3) modeText = "Mode 3: Cat Mask";

        cv::putText(frame, modeText,
                    cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0,255,0), 2);

        // ---- 5. 绘制人脸和关键点（复用你原来的代码）----
        if (!faces.empty()) {
            for (int i = 0; i < faces.rows; ++i) {
                int x1 = static_cast<int>(faces.at<float>(i, 0));
                int y1 = static_cast<int>(faces.at<float>(i, 1));
                int w  = static_cast<int>(faces.at<float>(i, 2));
                int h  = static_cast<int>(faces.at<float>(i, 3));

                cv::rectangle(frame, cv::Rect(x1, y1, w, h), cv::Scalar(0,255,0), 2);
        
                // 根据 mode 应用不同隐私处理
                cv::Rect faceRect(x1, y1, w, h);

                switch (mode) {
                    case 1:
                        filter.blurFace(frame, faceRect);
                        break;

                    case 2:
                        filter.pixelateFace(frame, faceRect);
                        break;

                    case 3:{
                        //使用猫猫做遮盖
                        //1.原始框
                        cv::Rect face(x1,y1,w,h);
                        //2.放大面部框
                        float scale = 1.5f;
                        int newH = h*scale;
                        int newW = w*scale;

                        int cx = x1+w/2;
                        int cy = y1+h/2;

                        cv::Rect bigFace(
                            cx - newW/2,
                            cy - newH/2,
                            newW,
                            newH
                        );

                        // 3. 限制在图像内
                        bigFace &= cv::Rect(0,0,frame.cols,frame.rows);


                        filter.maskFace(frame, bigFace, maskImg);
                        break;
                    }
                    default:
                        // mode == 0: 不做任何处理
                        break;
                }
                
                float conf = faces.at<float>(i, 14);
                cv::putText(frame, cv::format("Conf: %.2f", conf),
                            cv::Point(x1, y1 - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(0,255,0), 2);

                // 绘制五个关键点
                std::vector<cv::Scalar> colors{
                    cv::Scalar(255, 0, 0),
                    cv::Scalar(0, 0, 255),
                    cv::Scalar(0, 255, 0),
                    cv::Scalar(255, 0, 255),
                    cv::Scalar(0, 255, 255)
                };

                for (int j = 0; j < 5; ++j) {
                    int px = static_cast<int>(faces.at<float>(i, 4 + 2*j));
                    int py = static_cast<int>(faces.at<float>(i, 5 + 2*j));
                    cv::circle(frame, cv::Point(px, py), 3, colors[j], -1);
                }
            }
        }
        // ---- 6. 显示结果 ----
        cv::imshow("YuNet Face Detection", frame);

        // 退出逻辑, ESC or q
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q') break;
        if (key== '0') mode=0;
        if (key== '1') mode=1;
        if (key== '2') mode=2;
        if (key== '3') mode=3;

        tick.reset();
    }

    // ---- 7. 清理 ----
    cap.release();
    cv::destroyAllWindows();
    return 0;
}