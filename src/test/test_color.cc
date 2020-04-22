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

#include <vector>

#include <gtest/gtest.h> 

#include "core/color.h"

using namespace qjulia;

TEST(ColorConversion, RGB2XYZ2RGB1) {
  Vector3f rgb1(0.8, 0.1, 0.2);
  Vector3f xyz = RGB2XYZ(rgb1);
  Vector3f rgb2 = XYZ2RGB(xyz);
  EXPECT_NEAR(rgb1[0], rgb2[0], 1e-2);
  EXPECT_NEAR(rgb1[1], rgb2[1], 1e-2);
  EXPECT_NEAR(rgb1[2], rgb2[2], 1e-2);
}

TEST(ColorConversion, RGB2XYZ2RGB2) {
  Vector3f rgb1(0.5, 0.5, 0.5);
  Vector3f xyz = RGB2XYZ(rgb1);
  Vector3f rgb2 = XYZ2RGB(xyz);
  EXPECT_NEAR(rgb1[0], rgb2[0], 1e-2);
  EXPECT_NEAR(rgb1[1], rgb2[1], 1e-2);
  EXPECT_NEAR(rgb1[2], rgb2[2], 1e-2);
}

TEST(ColorConversion, RGB2Lab2RGB1) {
  Vector3f rgb1(0.5, 0.5, 0.5);
  Vector3f lab = RGB2Lab(rgb1);
  Vector3f rgb2 = Lab2RGB(lab);
  
  EXPECT_NEAR(lab[1], 0, 1e-2);
  EXPECT_NEAR(lab[2], 0, 1e-2);
  EXPECT_NEAR(rgb1[0], rgb2[0], 1e-2);
  EXPECT_NEAR(rgb1[1], rgb2[1], 1e-2);
  EXPECT_NEAR(rgb1[2], rgb2[2], 1e-2);
}

TEST(ColorConversion, RGB2Lab2RGB2) {
  Vector3f rgb1(0.2, 0.3, 0.5);
  Vector3f lab = RGB2Lab(rgb1);
  Vector3f rgb2 = Lab2RGB(lab);
  
  EXPECT_NEAR(rgb1[0], rgb2[0], 1e-2);
  EXPECT_NEAR(rgb1[1], rgb2[1], 1e-2);
  EXPECT_NEAR(rgb1[2], rgb2[2], 1e-2);
}

TEST(ColorConversion, RGB2LCH2RGB1) {
  Vector3f rgb1(0.8, 0.1, 0.2);
  Vector3f lch = RGB2LCH(rgb1);
  Vector3f rgb2 = LCH2RGB(lch);
  EXPECT_NEAR(rgb1[0], rgb2[0], 1e-2);
  EXPECT_NEAR(rgb1[1], rgb2[1], 1e-2);
  EXPECT_NEAR(rgb1[2], rgb2[2], 1e-2);
}

TEST(ColorConversion, RGB2LCH2RGB2) {
  Vector3f rgb1(0.2, 0.3, 0.5);
  Vector3f lch = RGB2LCH(rgb1);
  Vector3f rgb2 = LCH2RGB(lch);
  EXPECT_NEAR(rgb1[0], rgb2[0], 1e-2);
  EXPECT_NEAR(rgb1[1], rgb2[1], 1e-2);
  EXPECT_NEAR(rgb1[2], rgb2[2], 1e-2);
}
