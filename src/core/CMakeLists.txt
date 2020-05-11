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

configure_file(config.h.in config.h)

set(CPP_ONLY_SOURCE_FILES
  algorithm.cc
  image.cc
  vector.cc
  qjs_parser.cc
  image_io.cc
  image_io/image_io_png.cc
  image_io/image_io_tiff.cc
)

set(SHARED_SOURCE_FILES
  engine.cc
  film.cc
  material.cc
  camera.cc
  developer.cc
  dof_simulator.cc
  object.cc
  scene.cc
  world.cc
  scene_builder.cc
  ssaa.cc
  texture.cc
  transform.cc
  camera/camera3d.cc
  light/simple.cc
  shape/julia3d.cc
  shape/julia3d_kernel.cc
  shape/plane.cc
  shape/sphere.cc
  integrator/default.cc
)

add_library(qjulia2-core OBJECT
  ${SHARED_SOURCE_FILES}
  ${CPP_ONLY_SOURCE_FILES}
)

if (WITH_CUDA)
  set_source_files_properties(
    ${SHARED_SOURCE_FILES}
    PROPERTIES LANGUAGE CUDA
  )
endif()

set_target_properties(qjulia2-core
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(qjulia2-core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  "${CMAKE_CURRENT_BINARY_DIR}/.."
)

target_link_libraries(qjulia2-core
  fmt
  glog
  PNG::PNG
  ${TIFF_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
)
