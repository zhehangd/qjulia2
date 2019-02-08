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

#include "qjulia2/core/efloat.h"

TEST(EFloat, Regular) {
  qjulia::Float v = 3.14f;
  qjulia::BinaryFloat b = qjulia::FloatToBinary(v);
  EXPECT_EQ(b, 0x4048F5C3);
  EXPECT_EQ(qjulia::BinaryToFloat(b), v);
}

TEST(EFloat, PositiveZero) {
  qjulia::Float v = 0.f;
  qjulia::BinaryFloat b = qjulia::FloatToBinary(v);
  EXPECT_EQ(b, 0x00000000);
  EXPECT_EQ(qjulia::BinaryToFloat(b), v);
}

TEST(EFloat, NegativeZero) {
  qjulia::Float v = -0.f;
  qjulia::BinaryFloat b = qjulia::FloatToBinary(v);
  EXPECT_EQ(b, 0x80000000);
  EXPECT_EQ(qjulia::BinaryToFloat(b), v);
}

TEST(EFloat, SmallFraction) {
  qjulia::Float v = 1.00000012f;
  qjulia::BinaryFloat b = qjulia::FloatToBinary(v);
  EXPECT_EQ(b, 0x3F800001);
  EXPECT_EQ(qjulia::NextFloatDown(v), 1.0f);
  EXPECT_EQ(qjulia::BinaryToFloat(b), v);
  EXPECT_EQ(qjulia::NextFloatUp(qjulia::NextFloatDown(v)), v);
  EXPECT_EQ(qjulia::NextFloatDown(qjulia::NextFloatUp(v)), v);
}