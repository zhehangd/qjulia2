# MIT License
# 
# Copyright (c) 2019 Zhehang Ding
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.12)

set(WITH_CUDA OFF CACHE BOOL "Build GPU acceleration with CUDA")
set(WITH_GUI OFF CACHE BOOL "Build GUI based on Qt5")
set(WITH_TEST OFF CACHE BOOL "Build unit tests")
set(WITH_DOC OFF CACHE BOOL "Build doxygen documentation")

if (WITH_CUDA)
  message(STATUS "Build GPU acceleration with CUDA")
  set(LANGUAGE_SET CXX CUDA)
else()
  set(LANGUAGE_SET CXX)
endif()

project(qjulia2 VERSION 1.0.0 LANGUAGES ${LANGUAGE_SET})

if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
   message(FATAL_ERROR "In-source builds are not allowed.")
endif("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")

if(NOT CMAKE_BUILD_TYPE)
    message(WARNING "No build type selected. Default to Release.")
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules" ${CMAKE_MODULE_PATH})

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-pedantic>)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
else()
  message(SEND_ERROR "Unspported platform")
endif()

find_package(fmt REQUIRED)
find_package(glog REQUIRED)
find_package(Threads REQUIRED)

if (WITH_TEST)
  find_package(GTest REQUIRED)
endif()

if (WITH_GUI)
  message(STATUS "Build with GUI")
  find_package(Qt5 COMPONENTS Widgets Concurrent REQUIRED)
  set(CMAKE_AUTOMOC ON) # Instruct CMake to run moc automatically when needed
  set(CMAKE_AUTOUIC ON) # Create code from a list of Qt designer ui files
  find_package(OpenCV REQUIRED) # temp
endif()

add_subdirectory(3rdparty)
add_subdirectory(src)

if (WITH_DOC)
  add_subdirectory(doc)
endif()

file(COPY "${PROJECT_SOURCE_DIR}/data"
  DESTINATION ${PROJECT_BINARY_DIR})

