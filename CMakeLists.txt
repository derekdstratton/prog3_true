cmake_minimum_required(VERSION 3.17)
project(prog3)

set(CMAKE_CXX_STANDARD 17)

include_directories(eigen-3.3.9)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/x64/vc15/lib)

set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/x64/vc15/lib)

find_package(OpenCV CONFIG REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(prog3 main.cpp)

add_executable(prog3_exp experiments.cpp)

target_link_libraries( prog3 ${OpenCV_LIBS} )

target_link_libraries( prog3 ${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/x64/vc15/lib/opencv_world452d.lib)

target_link_libraries( prog3_exp ${OpenCV_LIBS} )

target_link_libraries( prog3_exp ${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/x64/vc15/lib/opencv_world452d.lib)

add_executable(pls retry.cpp)
target_link_libraries( pls ${OpenCV_LIBS} )

target_link_libraries( pls ${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/x64/vc15/lib/opencv_world452d.lib)


message("libs")
message(${OpenCV_LIBS})