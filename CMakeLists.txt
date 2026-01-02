cmake_minimum_required(VERSION 3.10)

# 项目名（可以叫 project，也可以叫 privacy_protector）
project(privacy_protector)

# 使用 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 找 OpenCV
find_package(OpenCV REQUIRED)

# 头文件搜索路径
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    include
)

# 可执行文件 + 源文件
add_executable(yunet
    src/main.cpp
    src/FaceDetector.cpp
    src/PrivacyFilter.cpp
)

# 链接 OpenCV
target_link_libraries(yunet
    ${OpenCV_LIBS}
)
