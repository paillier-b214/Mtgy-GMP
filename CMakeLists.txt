cmake_minimum_required(VERSION 3.22)
project(mtgy)

set(CMAKE_CXX_STANDARD 14)

add_executable(mtgy main.cpp)
target_link_libraries(mtgy gmp)
