cmake_minimum_required(VERSION 3.29)
project(machine_learning)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)

add_executable(neural_network src/neural_network.cc)
