cmake_minimum_required(VERSION 4.0)
project(micro_benchmarks)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")

add_executable(micro_benchmarks
        test_bench.cpp)
