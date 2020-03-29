/*

MIT License

Copyright (c) 2019 Zhehang Ding

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include <gtest/gtest.h> 

#include "core/scene_builder.h"
#include "core/camera/camera3d.h"
#include "core/light/simple.h"

using namespace qjulia;

TEST(SceneBuilder, General) {
  SceneBuilder scene_build;
  scene_build.Register<PerspectiveCamera>("Perspective");
  scene_build.Register<OrthoCamera>("Ortho");
  scene_build.Register<SunLight>("Sun");
  scene_build.Register<PointLight>("Point");
  
  auto *camera_node = scene_build.CreateEntity<Camera>("Perspective", "camera1");
  Camera *camera = camera_node->Get();
  EXPECT_EQ(camera_node->GetName(), "camera1");
  
  auto *light_node = scene_build.CreateEntity<Light>("Point", "light_01");
  Light *light = light_node->Get();
  EXPECT_EQ(light_node->GetName(), "light_01");
}

 
