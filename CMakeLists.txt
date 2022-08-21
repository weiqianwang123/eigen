cmake_minimum_required(VERSION 3.22)
project(20220716)

find_package(OpenCV 4 REQUIRED)
set(CMAKE_CXX_STANDARD 14)
INCLUDE_DIRECTORIES("/usr/include/eigen3")

add_executable(20220716 caculate.cpp)
target_link_libraries(20220716 PRIVATE ${OpenCV_LIBS})
