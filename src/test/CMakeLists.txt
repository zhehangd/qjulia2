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



add_executable(test_all
  test_main.cc
  test_array2d.cc
  test_texture.cc
  test_vector.cc
  test_algorithm.cc
  test_camera.cc
  test_scene_builder.cc
  test_color.cc
  test_args.cc
  test_transform.cc
  $<TARGET_OBJECTS:qjulia2-core>
)

if (WITH_CUDA)
  set_source_files_properties(
    test_texture.cc
    test_scene_builder.cc
    PROPERTIES LANGUAGE CUDA
  )
  
  set_target_properties(test_all
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
  )
endif()

target_include_directories(test_all PUBLIC ${GTEST_INCLUDE_DIRS})

target_link_libraries(test_all
  qjulia2-core
  glog
  ${GTEST_BOTH_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
  qjulia2-core
)
