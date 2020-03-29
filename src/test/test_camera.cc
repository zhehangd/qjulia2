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

#include "core/ray.h"
#include "core/vector.h"
#include "core/camera/camera3d.h"

using namespace qjulia;

TEST(Camera3D, LookAt) {
  PerspectiveCamera camera;
  camera.LookAt({1,2,3}, {0,0,5}, {0,1,0});
  Ray ray = camera.CastRay({0.5, 0.5});
  EXPECT_EQ(ray.start, Vector3f(1,2,3));
  EXPECT_LT(ray.dir[0], 0);
  EXPECT_LT(ray.dir[1], 0);
  EXPECT_GT(ray.dir[2], 0);
}

TEST(Camera3D, ParseLookAt) {
  PerspectiveCamera camera;
  camera.Parse({"LookAt", "1,2,3", "0,0,5", "0,1,0"}, nullptr);
  Ray ray = camera.CastRay({0.5, 0.5});
  EXPECT_EQ(ray.start, Vector3f(1,2,3));
  EXPECT_LT(ray.dir[0], 0);
  EXPECT_LT(ray.dir[1], 0);
  EXPECT_GT(ray.dir[2], 0);
}

