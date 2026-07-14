cmake_minimum_required(VERSION 3.16)
project(options_pricer CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(OPTIONSPRICER_BUILD_TESTS "Build unit tests" ON)

add_subdirectory(src)
add_subdirectory(apps)

if(OPTIONSPRICER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
