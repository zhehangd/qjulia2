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

add_library(qjulia2-core OBJECT
  algorithm.cc
  camera.cc
  efloat.cc
  engine.cc
  film.cc
  integrator.cc
  vector.cc
  light.cc
  scene.cc
  object.cc
  tokenizer.cc
  transform.cc
  material.cc
  messages.cc
  resource_mgr.cc
  camera/camera3d.cc
  integrator/default.cc
  integrator/normal.cc
  integrator/time.cc
  light/simple.cc
  shape/sphere.cc
  shape/plane.cc
  shape/julia3d.cc
)

target_include_directories(qjulia2-core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

target_link_libraries(qjulia2-core
  glog
  ${CMAKE_THREAD_LIBS_INIT}
)
