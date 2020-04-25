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

#include "core/transform.h"

using namespace qjulia;

TEST(DecomposeMatrix, Test1) {
  Matrix4x4 m = Matrix4x4::Translate({3, 7, 11})
    * Matrix4x4::RotateZ(42)
    * Matrix4x4::RotateY(30)
    * Matrix4x4::RotateX(15)
    * Matrix4x4::Scale({0.5, 0.9, 1.2});
  // T, R, S
  auto ret = DecomposeMatrix(m);
  EXPECT_NEAR(ret[0][0], 3, 1e-3);
  EXPECT_NEAR(ret[0][1], 7, 1e-3);
  EXPECT_NEAR(ret[0][2], 11, 1e-3);
  EXPECT_NEAR(ret[1][0], 15, 1e-3);
  EXPECT_NEAR(ret[1][1], 30, 1e-3);
  EXPECT_NEAR(ret[1][2], 42, 1e-3);
  EXPECT_NEAR(ret[2][0], 0.5, 1e-3);
  EXPECT_NEAR(ret[2][1], 0.9, 1e-3);
  EXPECT_NEAR(ret[2][2], 1.2, 1e-3);
}
