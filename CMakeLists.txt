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

cmake_minimum_required(VERSION 3.1)

project(qjulia2 VERSION 1.0.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  message(WARNING
    "No build type selected, will use Release by default. "
    "To use debug mode, configure cmake with -DCMAKE_BUILD_TYPE=Debug.")
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules" ${CMAKE_MODULE_PATH})
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

if (CMAKE_COMPILER_IS_GNUCC)
  add_compile_options(-Wall -pedantic -Wextra)
endif()


find_package(Threads REQUIRED)
find_package(GTest)

# ----------------------- 3rd party libraries ----------------------
add_subdirectory(3rdparty)

# ------------------------------- Main ------------------------------
add_subdirectory(src)

# ------------------------------- Doc -------------------------------
add_subdirectory(doc)

# ------------------------------- Test -------------------------------
if (${GTest_FOUND})
  add_subdirectory(test)
endif()

# ------------------------------- Data -------------------------------

file(COPY "${PROJECT_SOURCE_DIR}/data"
  DESTINATION ${PROJECT_BINARY_DIR})

