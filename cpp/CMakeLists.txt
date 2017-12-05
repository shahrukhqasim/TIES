cmake_minimum_required(VERSION 3.7)
project(cpp)

set(CMAKE_CXX_STANDARD 11)

link_directories(/usr/local/lib/ /usr/lib/ ${Boost_LIBRARY_DIR})
include_directories(/usr/local/include/ DocProc/src/ ${Boost_INCLUDE_DIR})

set(SOURCE_FILES main.cpp main_skew.cc)
add_executable(tessocr ${SOURCE_FILES})
add_executable(fixskew main_skew.cc)

target_link_libraries(tessocr lept tesseract opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs pthread gomp docproc)
target_link_libraries(fixskew lept tesseract opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs pthread gomp docproc)