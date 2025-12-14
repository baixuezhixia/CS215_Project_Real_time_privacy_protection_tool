cmake_minimum_required(VERSION 3.10)
project(project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    include 
)

file(GLOB SOURCES "src/*.cpp")

add_executable(project ${SOURCES})
target_link_libraries(project ${OpenCV_LIBS})

# 将 assets 文件夹复制到 build 目录
file(COPY assets DESTINATION ${CMAKE_BINARY_DIR})
